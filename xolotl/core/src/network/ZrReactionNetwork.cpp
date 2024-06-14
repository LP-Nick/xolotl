#include <xolotl/core/network/ZrReactionNetwork.h>
#include <xolotl/core/network/impl/ZrReactionNetwork.tpp>
#include <xolotl/util/MPIUtils.h>

namespace xolotl
{
namespace core
{
namespace network
{
template ReactionNetwork<ZrReactionNetwork>::ReactionNetwork(
	const std::vector<AmountType>& maxSpeciesAmounts,
	const std::vector<SubdivisionRatio>& subdivisionRatios, IndexType gridSize,
	const options::IOptions& opts);

template ReactionNetwork<ZrReactionNetwork>::ReactionNetwork(
	const std::vector<AmountType>& maxSpeciesAmounts, IndexType gridSize,
	const options::IOptions& opts);

template double
ReactionNetwork<ZrReactionNetwork>::getTotalConcentration(
	ConcentrationsView concentrations, Species type, AmountType minSize);

template double
ReactionNetwork<ZrReactionNetwork>::getTotalRadiusConcentration(
	ConcentrationsView concentrations, Species type, AmountType minSize);

template double
ReactionNetwork<ZrReactionNetwork>::getTotalAtomConcentration(
	ConcentrationsView concentrations, Species type, AmountType minSize);

template double
ReactionNetwork<ZrReactionNetwork>::getTotalTrappedAtomConcentration(
	ConcentrationsView concentrations, Species type, AmountType minSize);

template double
ReactionNetwork<ZrReactionNetwork>::getTotalVolumeFraction(
	ConcentrationsView concentrations, Species type, AmountType minSize);

double
ZrReactionNetwork::checkLatticeParameter(double latticeParameter)
{
	if (latticeParameter <= 0.0) {
		return alphaZrLatticeConstant;
	}
	return latticeParameter;
}

double
ZrReactionNetwork::checkImpurityRadius(double impurityRadius)
{
	if (impurityRadius <= 0.0) {
		return alphaZrCoreRadius;
	}
	return impurityRadius;
}

ZrReactionNetwork::IndexType
ZrReactionNetwork::checkLargestClusterId()
{
	// Copy the cluster data for the parallel loop
	auto clData = _clusterData.d_view;
	using Reducer = Kokkos::MaxLoc<ZrReactionNetwork::AmountType,
		ZrReactionNetwork::IndexType>;
	Reducer::value_type maxLoc;
	Kokkos::parallel_reduce(
		_numClusters,
		KOKKOS_LAMBDA(IndexType i, Reducer::value_type & update) {
			const Region& clReg = clData().getCluster(i).getRegion();
			Composition hi = clReg.getUpperLimitPoint();

			// adding basal
			auto size = hi[Species::V] + hi[Species::I] + hi[Species::Basal];

			if (size > update.val) {
				update.val = size;
				update.loc = i;
			}
		},
		Reducer(maxLoc));

	return maxLoc.loc;
}

void
ZrReactionNetwork::setConstantRates(RatesView rates, IndexType gridIndex)
{
	_reactions.forEachOn<ZrConstantReaction>(
		"ReactionCollection::setConstantRates", DEVICE_LAMBDA(auto&& reaction) {
			reaction.setRate(rates, gridIndex);
			reaction.updateRates();
		});
}

void
ZrReactionNetwork::setConstantConnectivities(ConnectivitiesPair conns)
{
	_constantConnsRows = ConnectivitiesPairView(
		"dConstantConnectivitiesRows", conns.first.size());
	_constantConnsEntries = ConnectivitiesPairView(
		"dConstantConnectivitiesEntries", conns.second.size());
	auto hConnsRowsView = create_mirror_view(_constantConnsRows);
	auto hConnsEntriesView = create_mirror_view(_constantConnsEntries);
	for (auto i = 0; i < conns.first.size(); i++) {
		hConnsRowsView(i) = conns.first[i];
	}
	for (auto i = 0; i < conns.second.size(); i++) {
		hConnsEntriesView(i) = conns.second[i];
	}
	deep_copy(_constantConnsRows, hConnsRowsView);
	deep_copy(_constantConnsEntries, hConnsEntriesView);
}

void
ZrReactionNetwork::setConstantRateEntries()
{
	auto rows = _constantConnsRows;
	auto entries = _constantConnsEntries;
	_reactions.forEachOn<ZrConstantReaction>(
		"ReactionCollection::setConstantRates", DEVICE_LAMBDA(auto&& reaction) {
			reaction.defineRateEntries(rows, entries);
		});
}

void
ZrReactionNetwork::setGridSize(IndexType gridSize)
{
	this->_clusterData.h_view().extraData.setGridSize(
		this->_clusterData.h_view().numClusters, gridSize);
	Superclass::setGridSize(gridSize);
}

void
ZrReactionNetwork::initializeExtraClusterData(const options::IOptions& options)
{
	this->_clusterData.h_view().extraData.initialize(
		this->_clusterData.h_view().numClusters,
		this->_clusterData.h_view().gridSize);
	this->copyClusterDataView();

	auto data = this->_clusterData.h_view();
	Kokkos::parallel_for(
		this->_numClusters, KOKKOS_LAMBDA(const IndexType i) {
			auto cluster = data.getCluster(i);
			const auto& reg = cluster.getRegion();
			Composition lo(reg.getOrigin());

			// Set the dislocation capture radii for vacancy a-loops (convert to
			// nm): First index in dislocation capture radius is for I capture;
			// second is for V capture
			if (lo.isOnAxis(Species::V)) {
				// Spontaneous radii:
				// data.extraData.dislocationCaptureRadius(i, 0) = 3.05 *
				// pow(lo[Species::V], 0.12) / 10;
				// data.extraData.dislocationCaptureRadius(i, 1) = 0.39 *
				// pow(lo[Species::V], 0.4) / 10;

				// Thermal radii:
				if (lo[Species::V] < 1000) {
					data.extraData.dislocationCaptureRadius(i, 0) =
						2.8 * pow(lo[Species::V], 0.15) / 10;
					data.extraData.dislocationCaptureRadius(i, 1) =
						2.0 * pow(lo[Species::V], 0.3) / 10;
				}
				else {
					data.extraData.dislocationCaptureRadius(i, 0) = 0.79;
					data.extraData.dislocationCaptureRadius(i, 1) = 1.59;
				}
			}

			// adding basal
			// Set the dislocation capture radii for vacancy c-loops (convert to
			// nm): First index in dislocation capture radius is for I capture;
			// second is for V capture
			if (lo.isOnAxis(Species::Basal)) {
				// Spontaneous radii:
				// if(lo[Species::Basal] < ::xolotl::core::basalTransitionSize)
				// data.extraData.dislocationCaptureRadius(i, 0) = 3.9 *
				// pow(lo[Species::Basal], 0.07) / 10; if(lo[Species::Basal] <
				// ::xolotl::core::basalTransitionSize)
				// data.extraData.dislocationCaptureRadius(i, 1) = 0.55 *
				// pow(lo[Species::Basal], 0.33) / 10;

				// Thermal radii:
				if (lo[Species::Basal] < 1000) {
					if (lo[Species::Basal] < data.transitionSize())
						data.extraData.dislocationCaptureRadius(i, 0) = 1.1;
					else
						data.extraData.dislocationCaptureRadius(i, 0) =
							5.2 * pow(lo[Species::Basal], 0.06) / 10;
					data.extraData.dislocationCaptureRadius(i, 1) =
						1.55 * pow(lo[Species::Basal], 0.28) / 10;
				}
				else {
					data.extraData.dislocationCaptureRadius(i, 0) = 0.787;
					data.extraData.dislocationCaptureRadius(i, 1) = 1.072;
				}
			}

			// Set the dislocation capture radii for interstitial a-loops
			// (convert to nm)
			else if (lo.isOnAxis(Species::I)) {
				// Spontaneous radii:
				// data.extraData.dislocationCaptureRadius(i, 0) = 4.2 *
				// pow(lo[Species::I], 0.05) / 10;
				// data.extraData.dislocationCaptureRadius(i, 1) = 5.1 *
				// pow(lo[Species::I], -0.01) / 10;

				// Thermal radii
				if (lo[Species::I] < 1000) {
					data.extraData.dislocationCaptureRadius(i, 0) =
						4.5 * pow(lo[Species::I], 0.205) / 10;
					data.extraData.dislocationCaptureRadius(i, 1) =
						6.0 * pow(lo[Species::I], 0.08) / 10;
				}
				else {
					data.extraData.dislocationCaptureRadius(i, 0) = 1.85;
					data.extraData.dislocationCaptureRadius(i, 1) = 1.04;
				}
			}
		}); // Goes with parallel_for
}

std::string
ZrReactionNetwork::getMonitorDataHeaderString() const
{
	std::stringstream header;

	auto numSpecies = getSpeciesListSize();
	header << "#time ";
	for (auto id = SpeciesId(numSpecies); id; ++id) {
		auto speciesName = this->getSpeciesName(id);
		header << speciesName << "_density " << speciesName << "_atom "
			   << speciesName << "_diameter " << speciesName
			   << "_partial_density " << speciesName << "_partial_atom "
			   << speciesName << "_partial_diameter ";
	}

	return header.str();
}

void
ZrReactionNetwork::addMonitorDataValues(Kokkos::View<const double*> conc,
	double fac, std::vector<double>& totalVals)
{
	auto numSpecies = getSpeciesListSize();
	const auto& minSizes = this->getMinRadiusSizes();
	for (auto id = SpeciesId(numSpecies); id; ++id) {
		using TQ = IReactionNetwork::TotalQuantity;
		using Q = TQ::Type;
		using TQA = util::Array<TQ, 6>;
		auto ms = minSizes[id()];
		auto totals = this->getTotals(conc,
			TQA{TQ{Q::total, id, 1}, TQ{Q::atom, id, 1}, TQ{Q::radius, id, 1},
				TQ{Q::total, id, ms}, TQ{Q::atom, id, ms},
				TQ{Q::radius, id, ms}});

		totalVals[(6 * id()) + 0] += totals[0] * fac;
		totalVals[(6 * id()) + 1] += totals[1] * fac;
		totalVals[(6 * id()) + 2] += totals[2] * 2.0 * fac;
		totalVals[(6 * id()) + 3] += totals[3] * fac;
		totalVals[(6 * id()) + 4] += totals[4] * fac;
		totalVals[(6 * id()) + 5] += totals[5] * 2.0 * fac;
	}
}

void
ZrReactionNetwork::writeMonitorDataLine(
	const std::vector<double>& localData, double time)
{
	auto numSpecies = getSpeciesListSize();

	// Sum all the concentrations through MPI reduce
	auto globalData = std::vector<double>(localData.size(), 0.0);
	MPI_Reduce(localData.data(), globalData.data(), localData.size(),
		MPI_DOUBLE, MPI_SUM, 0, util::getMPIComm());

	if (util::getMPIRank() == 0) {
		// Average the data
		for (auto i = 0; i < numSpecies; ++i) {
			auto id = [i](std::size_t n) { return 6 * i + n; };
			if (globalData[id(0)] > 1.0e-16) {
				globalData[id(2)] /= globalData[id(0)];
			}
			if (globalData[id(3)] > 1.0e-16) {
				globalData[id(5)] /= globalData[id(3)];
			}
		}

		// Set the output precision
		const int outputPrecision = 5;

		// Open the output file
		std::fstream outputFile;
		outputFile.open(
			getMonitorOutputFileName(), std::fstream::out | std::fstream::app);
		outputFile << std::setprecision(outputPrecision);

		// Output the data
		outputFile << time << " ";
		for (auto i = 0; i < numSpecies; ++i) {
			auto id = [i](std::size_t n) { return 6 * i + n; };
			outputFile << globalData[id(0)] << " " << globalData[id(1)] << " "
					   << globalData[id(2)] << " " << globalData[id(3)] << " "
					   << globalData[id(4)] << " " << globalData[id(5)] << " ";
		}
		outputFile << std::endl;

		// Close the output file
		outputFile.close();
	}
}

std::string
ZrReactionNetwork::getRxnDataHeaderString() const
{
	std::stringstream header;

	header << "# species "<<"conc "<<"table_1_0 "<<"table_2_0 "<<"table_3_0 "<<"table_4_0 "
				 <<"table_1_1 "<<"table_2_1 "<<"table_3_1 "<<"table_4_1 "
				 <<"table_1_2 "<<"table_2_2 "<<"table_3_2 "<<"table_4_2 "
				 <<"table_1_3 "<<"table_2_3 "<<"table_3_3 "<<"table_4_3 "
				 <<"table_1_4 "<<"table_2_4 "<<"table_3_4 "<<"table_4_4 "
				 <<"table_1_5 "<<"table_2_5 "<<"table_3_5 "<<"table_4_5 "
				 <<"table_1_6 "<<"table_2_6 "<<"table_3_6 "<<"table_4_6 "
				 <<"table_1_7 "<<"table_2_7 "<<"table_3_7 "<<"table_4_7 ";
				 
	//header << "test output to see if cluster ID bins are working";

	return header.str();
}

void
ZrReactionNetwork::addRxnDataValues(Kokkos::View<const double*> conc, 
				std::vector<std::vector<double>>& totalVals)
{
		
		auto data = this->_clusterData.h_view();
		std::vector<IndexType> vacMobile, vacImmobile, vacAloop, intMobile, intImmobile, intAloop, basalImmobile, cLoop;
		std::vector<std::vector<IndexType>> clusterBins;
		std::vector<double> binConcs (8, 0.0);
		std::vector<int> sizeThresholds {3,6,9,data.transitionSize()}; //upper size thresholds for mobile int, mobile vac, immobile vac and int, FBP
		/*Kokkos::parallel_for(
		this->_numClusters, KOKKOS_LAMBDA(const IndexType i) {*/
		
		for (auto i = 0; i<_numClusters;i++){
			
			auto cluster = data.getCluster(i);
			const auto& reg = cluster.getRegion();
			Composition lo(reg.getOrigin());
			
			if (lo.isOnAxis(Species::V) && lo[Species::V] <= sizeThresholds[1]) vacMobile.push_back(i);
			else if (lo.isOnAxis(Species::V) && lo[Species::V] <= sizeThresholds[2]) vacImmobile.push_back(i);
			else if (lo.isOnAxis(Species::V) && lo[Species::V] > sizeThresholds[2]) vacAloop.push_back(i);
			
			else if (lo.isOnAxis(Species::Basal) && lo[Species::Basal] < sizeThresholds[3]) basalImmobile.push_back(i);
			else if (lo.isOnAxis(Species::Basal) && lo[Species::Basal] >= sizeThresholds[3]) cLoop.push_back(i);
			
			else if (lo.isOnAxis(Species::I) && lo[Species::I] <= sizeThresholds[0]) intMobile.push_back(i);
			else if (lo.isOnAxis(Species::I) && lo[Species::I] <= sizeThresholds[2]) intImmobile.push_back(i);
			else if (lo.isOnAxis(Species::I) && lo[Species::I] > sizeThresholds[2]) intAloop.push_back(i);
			
			/*			
			totalVals[(8*i)+0] = (lo.isOnAxis(Species::V)) ? lo[Species::V] : 0;
			totalVals[(8*i)+1] = (lo.isOnAxis(Species::Basal)) ? lo[Species::Basal] : 0;
			totalVals[(8*i)+2] = (lo.isOnAxis(Species::I)) ? lo[Species::I] : 0;
			totalVals[(8*i)+3] = conc(i);
			totalVals[(8*i)+4] = getTableOne(conc, i, 0);
			totalVals[(8*i)+5] = getTableTwo(conc, i, 0);
			totalVals[(8*i)+6] = getTableThree(conc, i, 0);
			totalVals[(8*i)+7] = getTableFour(conc, i, 0);  */
		
	};
	
	clusterBins = {vacMobile, vacImmobile, vacAloop, intMobile, intImmobile, intAloop, basalImmobile, cLoop};
	for (auto i = 0;i<clusterBins.size();i++){
		for (auto j = 0;j<clusterBins[i].size();j++){
			binConcs[i] += conc(clusterBins[i][j]);
		}
	}
	
	
	auto tableOne = getTableOne(conc, clusterBins, 0); //8x8 vector of reaction rates for TableOne reaction 
	auto tableTwo = getTableTwo(conc, clusterBins, 0);
	auto tableThree = getTableOne(conc, clusterBins, 0);
	auto tableFour = getTableOne(conc, clusterBins, 0);
	
	for (auto i=0;i<totalVals.size();i++){
		totalVals[i][0] = i;
		totalVals[i][1] = binConcs[i];
		for (auto j = 0;j<tableOne[i].size();j++){
			totalVals[i][(4*j)+2] = tableOne[i][j];
			totalVals[i][(4*j)+3] = tableTwo[i][j];
			totalVals[i][(4*j)+4] = tableThree[i][j];
			totalVals[i][(4*j)+5] = tableFour[i][j]; 
			
		}
	} 

}

void
ZrReactionNetwork::writeRxnDataLine(
	const std::vector<std::vector<double>>& localData, double time)
{
	/*auto numSpecies = getSpeciesListSize();

	// Sum all the concentrations through MPI reduce
	auto globalData = std::vector<double>(localData.size(), 0.0);
	MPI_Reduce(localData.data(), globalData.data(), localData.size(),
		MPI_DOUBLE, MPI_SUM, 0, util::getMPIComm());

	if (util::getMPIRank() == 0) {
		// Average the data
		for (auto i = 0; i < numSpecies; ++i) {
			auto id = [i](std::size_t n) { return 6 * i + n; };
			if (globalData[id(0)] > 1.0e-16) {
				globalData[id(2)] /= globalData[id(0)];
			}
			if (globalData[id(3)] > 1.0e-16) {
				globalData[id(5)] /= globalData[id(3)];
			}
		}
	*/
		// Set the output precision
		const int outputPrecision = 5;

		// Open the output file
		std::fstream outputFile;
		outputFile.open(
			getRxnOutputFileName(), std::fstream::out | std::fstream::app);
		outputFile << std::setprecision(outputPrecision);

		// Output the data
		outputFile << "# time: "<<time << std::endl;
		/*for (auto i = 0; i < _numClusters; i++) {
			unsigned long int vSize = (localData[8*i+0]);
			unsigned long int bSize = (localData[8*i+1]);
			unsigned long int iSize = (localData[8*i+2]);
			outputFile << vSize << " " << bSize << " "
					   << iSize <<" "<< localData[8*i+3] << " "
					   << localData[8*i+4] << " "<< localData[8*i+5] << " "
						 << localData[8*i+6]<<" "<<localData[8*i+7]  << " "<< std::endl;
		}*/
		
		for (auto i = 0; i < localData.size(); i++){
			for (auto j = 0; j < localData[i].size(); j++){
				outputFile << localData[i][j] <<" ";
				//outputFile << localData[i].size();
			}
			outputFile << std::endl;
		}
		// Close the output file
		outputFile.close();
	
}

} // namespace network
} // namespace core
} // namespace xolotl
