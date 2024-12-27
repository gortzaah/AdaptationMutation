package Evolvability.Winter2023;

import HAL.GridsAndAgents.AgentSQ2Dunstackable;
import HAL.Gui.GridWindow;
import HAL.Gui.UIGrid;
import HAL.Gui.GifMaker;
import HAL.Tools.FileIO;

import static HAL.Util.*;

import HAL.GridsAndAgents.AgentGrid2D;
import HAL.Rand;
import HAL.Util;
import org.apache.commons.math3.distribution.BetaDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.Well19937c;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.*;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.JCommander;

//      //////////////////////////////  *****************   //////////////////////////////     //
//                                         ABM PARAMETERS                                      //
//      //////////////////////////////  *****************  //////////////////////////////      //


class ABMParameters implements Cloneable {
    // 1.  Set of initial condition parameters to reproduce the experiments depending on the phenotypic variance and the
    // deviation from the zero net growth.

    public ABMParameters() {
        mutationRates = Arrays.asList("both_1", "both_1e-1", "both_1e-2", "both_1e-3", "both_1e-4", "both_0");
    }

    @Parameter(names = "-mutationRates")
    public List<String> mutationRates;

    // The following initial parameters should be changed
    @Parameter(names = "-meanPdiv0", description = "mean probability of division at t=0")
    public double MeanPdiv0 = 0.255; //Default: 0.255; just above the zero growth value (0.25)
    @Parameter(names = "-stDevPdiv0", description = "phenotypic variance: standard deviation for probability of division; equal for all cells")
    public double StDevPdiv0 = 0.001;   // Default: 0.001;
    @Parameter(names = "-stDevPdie0", description = "phenotypic variance: standard deviation for probability of death; equal for all cells")
    public double StDevPdie0 = 0.001;   // Default: 0.001;

    //Initial condiitions that should not be changing; kept at default values
    @Parameter(names = "-meanPdie0", description = "mean probability of death at t = 0")
    public double MeanPdie0 = 0.20;
    @Parameter(names = "-maxPdiv ")
    public double MaxPdiv = 1;
    @Parameter(names = "-minPdiv")
    public double MinPdiv = 0;
    @Parameter(names = "-maxPdie")
    public double MaxPdie = 1;
    @Parameter(names = "-minPdie")
    public double MinPdie = 0.196; //equivalent to mean_pdie0 - 4 * StDevPdiv0 with StDevPdiv0 = 0.001; //


    //2. Set of parameters to control initial tumor geometry
    // Default: circular geometry -- "Circular", "RandomScatter" otherwise

    // TODO: fill in valid values into the description and add a validator
    @Parameter(names = "-initialGeometryChoice")
    public String InitialGeometryChoice = "Circular"; //"RandomScatter";//"Circular";
    //The following parameters  depend on the initial tumor geometry;
    //radius = 4 gives 49 = 4^2 * pi - 1 cells
    @Parameter(names = "-intialNoCells")
    public int InitialNoCells = 49; ///49; // ONLY for random scatter initial geometry
    @Parameter(names = "-initialRadius")
    public double InitialRadius = 4;//17.84; //4; //39.9;//12.63; //4; // ONLY for circular initial geometry

    //3. Control Beta distribution
    @Parameter(names = "-lowerBoundBeta")
    public double LowerBoundBeta = 0.0833333332; //corresponding to ~0.5% of non-neutral mutations in a N(0.5,0.33)
    @Parameter(names = "-upperBoundBeta")
    public double UpperBoundBeta = 0.9166666668; //corresponding to ~0.5% of non-neutral mutations in a N(0.5,0.33)
    @Parameter(names = "-alphaBeta")
    public double alphaBeta = 3; // alpha parameter in beta(alpha,beta) distribution
    @Parameter(names = "-betaBeta")
    public double betaBeta = 3; // beta  parameter in beta(alpha,beta) distribution

    //4. Grid and simulation parameters
    @Parameter(names = "-domainSideLength")
    public int DomainSideLength = 100;   // Domain Side Length
    @Parameter(names = "-useVisualization", arity = 1)
    public boolean UseVisualization = false; // if true, remember to close all the windows!
    @Parameter(names = "-averageTimeStep")
    public int averageTimeStep = 100;
    @Parameter(names = "-iSCTimeStep")
    public int iSCTimeStep = 25;
    @Parameter(names = "-deadCellsTimeStep")
    public int deadCellsTimeStep = 5500;
    @Parameter(names = "-totalSimulationTime")
    public int TotalSimulationTime = 5500; //40000; //for the trajectories for the random scatter //  5500;


    @Parameter(names = "-maxRepetitions", description = "Max average replicates per parameter/mutation set. Takes precendence over the MutationRun replicates. Set to 0 for infinite.")
    public int maxRepetitions = 0;

    //5. Directory determination
    @Parameter(names = "-outputPrefix")
    //TODO change here
    public String outputPrefix = "cat_may18/";

    @Parameter(names = "-help", help = true)
    public boolean help;

    @Parameter(names = "-threads", description = "Number of threads to execute on. Should be close to the number of cores. 0 disables thread pool.")
    public int nThreads = 0;

    @Parameter
    public String outputDirectory = "/home/gosia/data/output/";
            //"/Volumes/PhD_Data_MW/";
    //String outputDirectory = "/Users/80021045/Dropbox/PhD project/HAL-Results/";  //"/Volumes/PhD_Data_MW/average_data_population/";
    //public ;  //""/Users/80021045/Dropbox/PhD project/HAL-Results/Representative_movies/";
    //public String outputDirectory = ;  //""/Users/80021045/Dropbox/PhD project/HAL-Results/Representative_movies/";

    // Creates a copy of this object
    public ABMParameters clone() throws CloneNotSupportedException {
        return (ABMParameters) super.clone();
    }

    public static ThreadPoolExecutor executor = null;
}

class IterationParameters {
    public IterationParameters() {
        //Iterate over mean division parameters and values for phenotypic variance.
        meanPDivs = List.of(0.255); // Arrays.asList(0.15, 0.20, 0.24, 0.25, 0.26, 0.3, 0.35);
        stddevPdivPdie = List.of(0.001); //List.of(0.001);
    }

    @Parameter(names = "-pDivs")
    public List<Double> meanPDivs;

    @Parameter(names = "-stddevPDivPDie")
    public List<Double> stddevPdivPdie;
}

//      //////////////////////////////  *****************   //////////////////////////////     //
//                                   CELL CLASS DEFINITION                                     //
//      //////////////////////////////  *****************  //////////////////////////////      //

class Cell extends AgentSQ2Dunstackable<TumorEvolution> {

    double MeanPdiv; // Mean probability of division of a cell
    double MeanPdie; // Mean probability of death of a cell
    double EffectivePdiv; // Effective probability of division, sampled from Normal distribution with mean = MeanPdiv,std_dev = G.init_conds.std_pdiv0
    double EffectivePdie; // Effective probability of death, sampled from Normal distribution with mean = MeanPdie,std_dev = G.init_conds.std_pdiv0
    double DivisionDependentPmut;
    double DivisionIndependentPmut;
    int CountNeutralMutations = 0;
    int CountPositiveMutations = 0;
    int CountNegativeMutations = 0;
    double AccumulatedMutationImpact;
    double PresentPositiveMutationImpact;
    double PresentNegativeMutationImpact;
    double PastAccumulatedMutationImpact = 0;
    int LineageID;
    boolean isDividing = false;
    double Pspace;


    //////////////////////////////////////// DEFINE ALL FUNCTIONS NECESSARY TO ARRANGE THE ABM //////////////////////////////////////////////


    // Color determination: pixel color = cell fitness
    void Draw() {

        assert G != null;
        if (G.vis == null) {
            return;
        }
        double fitness = MeanPdiv - MeanPdie;
        if (!IsAlive()) {
            G.vis.SetPix(Isq(), BLACK);
        } else if (fitness < 0.1) {
            G.vis.SetPix(Isq(), Util.RGB256(135, 206, 235));
        } else {
            G.vis.SetPix(Isq(), Util.HeatMapRGB(fitness));
        }


    }

    // What is the probability that cell has space to divide?
    void calculateFreeSpaceProbability() {
        int NumberOfFreeSpots = MapEmptyHood(G.neighborhood);
        int NumberOfOccupiedSpots = MapOccupiedHood(G.neighborhood);
        int TotalSpots = NumberOfFreeSpots + NumberOfOccupiedSpots;
        Pspace = ((double) NumberOfFreeSpots / TotalSpots);
    }

    //Sample from Normal Distribution with constant standard deviation and mean = cell's probability of division
    public double DetermineEffectivePdiv() {
        double EffectivePdiv = G.rn.Gaussian(MeanPdiv, G.abmParameters.StDevPdiv0);
        assert G.abmParameters.MaxPdiv > G.abmParameters.MinPdiv;
        EffectivePdiv = Math.min(EffectivePdiv, G.abmParameters.MaxPdiv);
        EffectivePdiv = Math.max(EffectivePdiv, G.abmParameters.MinPdiv);
        return EffectivePdiv;
    }

    //Sample from Normal Distribution with constant standard deviation and mean = cell's probability of death
    public double DetermineEffectivePdie() {
        double EffectivePdie = G.rn.Gaussian(MeanPdie, G.abmParameters.StDevPdie0);
        assert G.abmParameters.MaxPdie > G.abmParameters.MinPdie;
        EffectivePdie = Math.min(EffectivePdie, G.abmParameters.MaxPdie);
        EffectivePdie = Math.max(EffectivePdie, G.abmParameters.MinPdie);
        return EffectivePdie;
    }

    //If mutation is positive (beneficial to cell fitness), apply it to either pdiv or pdie; stay within the boundary
    double Calculate_Positive_Effect_Of_Mutation(double effect, boolean is_pdiv_affected) {
        double set_new_pdiv_or_pdie;

        if (is_pdiv_affected) {
            //INCREASE PDIV AND MAKE SURE TO STAY WITHIN THE BOUNDARY
            if (((effect + MeanPdiv) > G.abmParameters.MaxPdiv) || (MeanPdiv > G.abmParameters.MaxPdiv)) {
                set_new_pdiv_or_pdie = G.abmParameters.MaxPdiv;
            } else {
                set_new_pdiv_or_pdie = MeanPdiv + effect;
            }
        } else {
            // DECREASE PDIE AND MAKE SURE TO STAY WITHIN THE BOUNDARY
            if (((MeanPdie - effect) < G.abmParameters.MinPdie) || (MeanPdie < G.abmParameters.MinPdie)) {
                set_new_pdiv_or_pdie = G.abmParameters.MinPdie;
            } else {
                set_new_pdiv_or_pdie = MeanPdie - effect;
            }
        }
        return set_new_pdiv_or_pdie;
    }


    //If mutation is negative (deleterios to cell fitness), apply it to either pdiv or pdie; stay within the boundary
    double Calculate_Negative_Effect_Of_Mutation(double effect, boolean is_pdiv_affected) {
        double set_new_pdiv_or_pdie = 0;

        //DECREASE PDIV AND MAKE SRUE TO STAY WITHIN THE BOUNDARY
        if (is_pdiv_affected) {
            if (((MeanPdiv - effect) < G.abmParameters.MinPdiv) || (MeanPdiv < G.abmParameters.MinPdiv)) {
                set_new_pdiv_or_pdie = G.abmParameters.MinPdiv;
            } else {
                set_new_pdiv_or_pdie = MeanPdiv - effect;
            }
        } else {
            //INCREASE PDIE AND MAKE SURE TO STAY WITHIN THE BOUNDARY
            if (((MeanPdie + effect) > G.abmParameters.MaxPdie) || (MeanPdie > G.abmParameters.MaxPdie)) {
                set_new_pdiv_or_pdie = G.abmParameters.MaxPdie;
            } else {
                set_new_pdiv_or_pdie = MeanPdie + effect;
            }
        }
        return set_new_pdiv_or_pdie;
    }

    // GENERATE A BETA-DISTRIBUTED NUMBER  FOR MaybeMutate()
    // Beta distribution implemented from org.apache.commons.math3.distribution.BetaDistribution
    public static double nextBeta(double alpha, double beta, RandomGenerator rng) {
        return new BetaDistribution(rng, alpha, beta, BetaDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY).sample();
    }

    // CALCULATE MUTATIONAL IMPACT FOR MaybeMutate()
    double CalculateMutationImpact() {
        double MutationImpact;
        MutationImpact = nextBeta(G.abmParameters.alphaBeta, G.abmParameters.betaBeta, G.ReplicateRNfromRNG);
        return MutationImpact;
    }


    //////////////////////////////////////// ACTIONS OF THE ABM //////////////////////////////////////////////


    // Decision: Cell death?
    boolean MaybeDie() {
        if (!IsAlive())
            return false;

        EffectivePdie = DetermineEffectivePdie();
       // MeanPdie = EffectivePdie;                                           //TODO plasticty
        if (G.rn.Double() <= EffectivePdie) {
            Dispose();
            return true;
        }
        return false;
    }


    //Decision: Cell mutation? If yes, determine the impact.
    void MaybeMutate(double mutation_probability) {
        if (!IsAlive()) {
            return;
        }


        if (G.rn.Double() < mutation_probability) {

            //Generate a B(alpha,beta) number to determine the effect of the mutation
            double MutationImpact = CalculateMutationImpact(); //mutation impact from beta distribution
            double TransformedMutationImpact; // MutationImpact transformed to [0,1] interval

            // NEGATIVE IMPACT
            if (MutationImpact <= G.abmParameters.LowerBoundBeta) {
                CountNegativeMutations = CountNegativeMutations + 1;

                TransformedMutationImpact = ((G.abmParameters.LowerBoundBeta - MutationImpact) / G.abmParameters.LowerBoundBeta);

                // 50% chance to affect either MeanPdiv or MeanPdie
                if (G.rn.Double() < 0.5) {
                    MeanPdiv = Calculate_Negative_Effect_Of_Mutation(TransformedMutationImpact, true);
                } else {
                    MeanPdie = Calculate_Negative_Effect_Of_Mutation(TransformedMutationImpact, false);
                }

                AccumulatedMutationImpact = PastAccumulatedMutationImpact - TransformedMutationImpact;

                //POSITIVE IMPACT
            } else if (MutationImpact >= G.abmParameters.UpperBoundBeta) {
                CountPositiveMutations = CountPositiveMutations + 1;

                TransformedMutationImpact = ((MutationImpact - G.abmParameters.UpperBoundBeta) / (1 - G.abmParameters.UpperBoundBeta));

                // 50% chance to affect either MeanPdiv or MeanPdie
                if (G.rn.Double() < 0.5) {
                    MeanPdiv = Calculate_Positive_Effect_Of_Mutation(TransformedMutationImpact, true);
                } else {
                    MeanPdie = Calculate_Positive_Effect_Of_Mutation(TransformedMutationImpact, false);
                }
                AccumulatedMutationImpact = PastAccumulatedMutationImpact + TransformedMutationImpact;

                // NEUTRAL IMPACT
            } else {
                CountNeutralMutations = CountNeutralMutations + 1;
                AccumulatedMutationImpact = PastAccumulatedMutationImpact;
            }

            //Accumulate mutational impact
            PastAccumulatedMutationImpact = AccumulatedMutationImpact;

            // If the accumulated effect is positive, save it as a positive effect
            if (AccumulatedMutationImpact > 0) {
                PresentPositiveMutationImpact = AccumulatedMutationImpact;
                PresentNegativeMutationImpact = 0;
            }
            // If the accumulated effect is negative, save it as a negative effect
            if (AccumulatedMutationImpact < 0) {
                PresentNegativeMutationImpact = AccumulatedMutationImpact;
                PresentPositiveMutationImpact = 0;
            }
        }
    }


    //Decision: Cell division (+ eventual mutation)?
    void MaybeDivide() {

        // CHECK THE NEIGHBORHOOD AND PROBABILITY OF DIVISION
        int NumEmptySpaces = MapEmptyHood(G.neighborhood);

        EffectivePdiv = DetermineEffectivePdiv();
      //  MeanPdiv = EffectivePdiv;                                                               //TODO plasticty
        if ((G.rn.Double() <= EffectivePdiv) && NumEmptySpaces > 0) {
            isDividing = true;

            //Create Daughter cell
            int DaughterPosition = G.neighborhood[G.rn.Int(NumEmptySpaces)];
            Cell Daughter = G.NewAgentSQ(DaughterPosition);

            //Make sure the Daughter cells inherits everything from its parent
            Daughter.MeanPdie = MeanPdie;
            Daughter.MeanPdiv = MeanPdiv;
            Daughter.DivisionDependentPmut = DivisionDependentPmut;
            Daughter.DivisionIndependentPmut = DivisionIndependentPmut;
            Daughter.AccumulatedMutationImpact = AccumulatedMutationImpact;
            Daughter.LineageID = LineageID;
            Daughter.CountNegativeMutations = CountNegativeMutations;
            Daughter.CountPositiveMutations = CountPositiveMutations;
            Daughter.CountNeutralMutations = CountNeutralMutations;

            //Mother and daughter cells might mutate
            MaybeMutate(DivisionDependentPmut);
            Daughter.MaybeMutate(DivisionDependentPmut);
        } else {
            isDividing = false;
        }
    }
}


//      //////////////////////////////  *****************   //////////////////////////////     //
//               TISSUE (TUMOR EVOLUTION) CLASS DEFINITION: INITIALIZATION                     //
//      //////////////////////////////  *****************  //////////////////////////////      //

public class TumorEvolution extends AgentGrid2D<Cell> {


    //Initialize ABM parameters
    ABMParameters abmParameters;
    int deadCellsTimeStep;

    //Document progress: initialize cell files and counts
    FileIO averagesFile;
    FileIO cellDataFile;
    FileIO deadCellDataFile;

    public int TotalCellCount = 0;
    public double TotalPdie = 0;
    public double TotalPdiv = 0;
    public int TotalNeutralMutationsCount = 0;
    public int TotalNegativeMutationsCount = 0;
    public int TotalPositiveMutationsCount = 0;
    public double TotalPositiveImpact = 0;
    public double TotalNegativeImpact = 0;
    public int ActivelyDividingCellsCount = 0;
    public double TotalFreeSpots = 0;
    public boolean areAllDead = false; // Stop recording if all cells are dead

    //Random number generation
    Rand rn;
    RandomGenerator ReplicateRNfromRNG;

    //Space initialization: neighborhood and visualtization
    public int[] neighborhood = MooreHood(false);
    UIGrid vis;


    //Initial probability of division, initital probabilities of death and initial tumor geometry
    double SetInitialPdiv(double MeanPdiv0, double StDevPdiv0) {
        double Pdiv0 = Math.abs(rn.Gaussian(MeanPdiv0, StDevPdiv0));
        Pdiv0 = Math.min(Pdiv0, abmParameters.MaxPdiv);
        Pdiv0 = Math.max(Pdiv0, abmParameters.MinPdiv);
        return Pdiv0;
    }

    double SetInitialPdie(double MeanPdie0, double StDevPdie0) {
        double Pdie0 = Math.abs(rn.Gaussian(MeanPdie0, StDevPdie0));
        Pdie0 = Math.min(Pdie0, abmParameters.MaxPdie);
        Pdie0 = Math.max(Pdie0, abmParameters.MinPdie);
        return Pdie0;
    }

    void DetermineInitialGeometry(int InitialNoCells, double InitialRadius, int sideLength, double DivDependPmut, double DivIndependPmut) {
        String CircularGeometry = "Circular";
        String RandomScatterGeometry = "RandomScatter";

        assert abmParameters.InitialGeometryChoice != null;

        if (abmParameters.InitialGeometryChoice.equalsIgnoreCase(RandomScatterGeometry)) {
            int CellCount0 = InitialNoCells;
            int i = 0;
            while (i < CellCount0) {
                int x = rn.Int(sideLength);
                int y = rn.Int(sideLength);
                //avoid having two agents at the same position
                if (PopAt(x, y) != 0) {
                    continue;
                }
                Cell cell = NewAgentSQ(x, y);
                double Pdiv0 = SetInitialPdiv(abmParameters.MeanPdiv0, abmParameters.StDevPdiv0);
                double Pdie0 = SetInitialPdie(abmParameters.MeanPdie0, abmParameters.StDevPdie0);
                cell.MeanPdiv = Pdiv0;
                cell.MeanPdie = Pdie0;
                cell.DivisionDependentPmut = DivDependPmut;
                cell.DivisionIndependentPmut = DivIndependPmut;
                i++;
                cell.LineageID = i;
            }
        }

        if (abmParameters.InitialGeometryChoice.equalsIgnoreCase(CircularGeometry)) {
            int[] Coordinates = CircleHood(true, InitialRadius);
            int CellCount0 = MapHood(Coordinates, sideLength / 2, sideLength / 2);
            for (int i = 0; i < CellCount0; i++) {
                Cell cell = NewAgentSQ(Coordinates[i]);
                double Pdiv0 = SetInitialPdiv(abmParameters.MeanPdiv0, abmParameters.StDevPdiv0);
                double Pdie0 = SetInitialPdie(abmParameters.MeanPdie0, abmParameters.StDevPdie0);
                cell.MeanPdiv = Pdiv0;
                cell.MeanPdie = Pdie0;
                cell.DivisionDependentPmut = DivDependPmut;
                cell.DivisionIndependentPmut = DivIndependPmut;
                cell.LineageID = i;
            }
        }

    }


    // constructor -- initialize the whole tumor and files to save the data
    TumorEvolution(int sideLength, String averagesFileName1, String cellDataFileName, String deadCellDataFileName, int seed, UIGrid vis, double DivDependPmut, double DivIndependPmut,
                   ABMParameters ABMparameters) {


        //Super constructor, visualization, random number and rnd number generator
        super(sideLength, sideLength, Cell.class, false, false);
        this.vis = vis;
        rn = new Rand(seed);
        ReplicateRNfromRNG = new Well19937c(rn.Long(Long.MAX_VALUE));
        this.abmParameters = ABMparameters;
        deadCellsTimeStep = abmParameters.deadCellsTimeStep;

        // "seed" clone for common ancestry (non-existent in simulation)

        // initialize the data output files
        averagesFile = new FileIO(averagesFileName1, "w");
        cellDataFile = new FileIO(cellDataFileName, "w");
        deadCellDataFile = new FileIO(deadCellDataFileName, "w");

        // Determine the initial geometry
        DetermineInitialGeometry(abmParameters.InitialNoCells, abmParameters.InitialRadius, sideLength, DivDependPmut, DivIndependPmut);
    }


    // Step function ("steps" all cells through birth/death/mutation)
    void Step(int time, double DivisionDependentPmut, double DivisionIndependentPmut) {
        for (Cell c : this) {
            c.calculateFreeSpaceProbability();

            c.DivisionDependentPmut = DivisionDependentPmut;
            c.DivisionIndependentPmut = DivisionIndependentPmut;

            //Update cell-specific death, division and mutation rates
            if (c.MaybeDie()) {
                WriteDeadCellData(c, time, abmParameters.deadCellsTimeStep);
                if (vis != null) {
                    vis.SetPix(c.Isq(), Util.BLACK);
                }
            } else {
                c.MaybeMutate(c.DivisionIndependentPmut);
                c.MaybeDivide();
            }
            c.Draw();
        }
        CleanShuffle(rn);
    }


//      //////////////////////////////  *****************   //////////////////////////////     //
//                                             MAIN                                            //
//      //////////////////////////////  *****************  //////////////////////////////      //

public static void main(String[] args)
        throws CloneNotSupportedException, InterruptedException, IOException, InvalidCommandLineException {

        ABMParameters baseAbmParameters = new ABMParameters();
        IterationParameters iterationParameters = new IterationParameters();

        JCommander jCommander = JCommander.newBuilder()
                .addObject(baseAbmParameters)
                .addObject(iterationParameters)
                .build();
        jCommander.parse(args);
        if (baseAbmParameters.help) {
            jCommander.usage();
            return;
        }

        if (baseAbmParameters.nThreads > 0) {
            ABMParameters.executor = new ThreadPoolExecutor(baseAbmParameters.nThreads, baseAbmParameters.nThreads,
                    1, TimeUnit.SECONDS, new LinkedBlockingQueue<Runnable>());
        }

        //iterate over different std deviatons and pdiv0;
        for (double current_StDevPdivPdie : iterationParameters.stddevPdivPdie) {
            for (double current_MeanPdiv0 : iterationParameters.meanPDivs) {
                ABMParameters abmParameters = baseAbmParameters.clone();
                abmParameters.MeanPdiv0 = current_MeanPdiv0;
                abmParameters.StDevPdiv0 = current_StDevPdivPdie;
                abmParameters.StDevPdie0 = current_StDevPdivPdie;

                RunWithParameters(abmParameters);
            }
        }

        if (baseAbmParameters.nThreads > 0) {
            ABMParameters.executor.shutdown();

            while (!ABMParameters.executor.awaitTermination(5, TimeUnit.SECONDS)) {
                System.out.println(
                        "Waiting for task completion, " + ABMParameters.executor.getActiveCount() + " active, " +
                                ABMParameters.executor.getCompletedTaskCount() + " completed, " +
                                ABMParameters.executor.getQueue().size() + " pending.");
            }
        }
    }

    public static String outputFilePrefixFromInitialConditions(ABMParameters abmParameters) { //
        return String.format("%s/std=%.3f/pdiv0=%.3f/", abmParameters.outputPrefix, abmParameters.StDevPdiv0, abmParameters.MeanPdiv0);
    }

    public static void RunWithParameters(ABMParameters abmParameters) throws IOException, InvalidCommandLineException {
        //Initialize simulation and data-writing parameters
        int DomainSideLength = abmParameters.DomainSideLength;            // domain size
        boolean UseVisualization = abmParameters.UseVisualization;        // do you want to see the movies?
        GridWindow vis = null;
        String outputDirectory = abmParameters.outputDirectory;
        if (!outputDirectory.endsWith("/")) {
            outputDirectory += "/";
        }
        String outputPrefixFromInitialConditions = outputFilePrefixFromInitialConditions(abmParameters);
        String combinedOutputPrefix = outputDirectory + outputPrefixFromInitialConditions;

        if (UseVisualization && abmParameters.nThreads > 0) {
            throw new InvalidCommandLineException("Can't have UseVisualization AND threading.");
            // I guess technically we could I just don't want to deal with the headache.
        }

        List<MutationRun> mutationRuns = getMutationRuns(abmParameters);
        // Run simulations for all mutation rates in the pre-defined experimental set (different mutation rates)
        for (MutationRun m : mutationRuns) {

            //Initialize the domain, run and save the data
            if (UseVisualization) {
                vis = new GridWindow(DomainSideLength, DomainSideLength, 2);
            }

            // Set iterations over multiple replicates
            // TODO: The below should be "seed < TotalAveragesReplicates + SEED_START"
            //       to be independent of the actual SEED_START.
            //       DO NOT MODIFY SEED_START until that is fixed :)
            int SEED_START = 1;
            for (int seed = SEED_START; seed <= m.TotalAverageReplicates(); seed++) {
                if (abmParameters.maxRepetitions > 0 && seed > abmParameters.maxRepetitions) {
                    break;
                }

                if (abmParameters.nThreads > 0) {
                    int finalSeed = seed;
                    ABMParameters.executor.submit(() -> {
                        try {
                            runMutationRun(abmParameters, finalSeed, m, DomainSideLength, null, combinedOutputPrefix);
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    });
                } else {
                    runMutationRun(abmParameters, seed, m, DomainSideLength, vis, combinedOutputPrefix);
                }
            } // this is where the replicate loop ends

        } //end of the iteration over all mutation rates

//        if (vis != null) {
//            //vis.Close();
//        }
    }

    // Get enabled mutation runs, add parameter overrides if given.
    protected static List<MutationRun> getMutationRuns(ABMParameters abmParameters) throws InvalidCommandLineException {
        List<MutationRun> mutationRuns = new ArrayList<MutationRun>();
        Set<String> seen = new HashSet<String>();
        for (String enabledRateNames : abmParameters.mutationRates) {
            String name = enabledRateNames;
            String parameterOverrides = "";
            if (enabledRateNames.contains(":")) {
                String[] parts = enabledRateNames.split(":");
                if (parts.length != 2) {
                    throw new InvalidCommandLineException(
                            "Got parameter string " + enabledRateNames + " with invalid format (need exactly 2 parts)");
                }
                name = parts[0];
                parameterOverrides = parts[1];
            }
            
            if (seen.contains(name)) {
                throw new InvalidCommandLineException("MutationRun " + name + " specified multiple times.");
            }
            seen.add(name);

            MutationRun run = null;
            for (MutationRun m : MutationRun.values()) {
                if (m.MutationNameDefinition().equals(name)) {
                    run = m;
                    break;
                }
            }
            if (run == null) {
                throw new InvalidCommandLineException(
                        "Unable to find mutationrun with name " + name + ", from parameter string " + enabledRateNames);
            }
            if (parameterOverrides.length() > 0) {
                run.UpdateFromParameterString(parameterOverrides);
            }
            mutationRuns.add(run);
        }
        return mutationRuns;
    }

    private static void runMutationRun(ABMParameters abmParameters, int seed, MutationRun m, int DomainSideLength, GridWindow vis, String combinedOutputPrefix) throws IOException {
        double DivisionIndependentPmut = m.DivisionIndependentPMut();
        double DivisionDependentPmut = m.DivisionDependentPMut();
        int TotaliSCReplicates = m.TotaliSCReplicates();
        int averageTimeStep = abmParameters.averageTimeStep;              // when to save average data (saves when time % modifier == 0)
        int iSCTimeStep = abmParameters.iSCTimeStep;                      // when to save iSC data
        int totalSimulationTime = abmParameters.TotalSimulationTime;      // total simulation time

        String OutputFilePrefix_all = combinedOutputPrefix + m.MutationNameDefinition() + "/";


        if (abmParameters.nThreads == 0) {
            System.out.println("Seed: " + seed);
        }
        try {
            Files.createDirectories(Paths.get(OutputFilePrefix_all));
            Files.createDirectories(Paths.get(OutputFilePrefix_all));
        } catch (IOException e) {
            e.printStackTrace();
            throw e;
        }

        //Pre-define data names
        String averagesFileName = OutputFilePrefix_all + "averages_" + seed + ".csv";
        String iSCFileName = OutputFilePrefix_all + "mutations_" + seed + ".csv";
        String deadCellsFileName = OutputFilePrefix_all + "dead_" + seed + ".csv";

        //Create the tumor
        TumorEvolution tumorEvolution = new TumorEvolution(DomainSideLength, averagesFileName, iSCFileName, deadCellsFileName,
                seed, vis, DivisionDependentPmut, DivisionIndependentPmut, abmParameters);
        GifMaker gif = null;

        if (vis != null) {
            gif = new GifMaker(OutputFilePrefix_all + "gif_no_" + seed + ".gif", 100, true);
            vis.Clear(BLACK);
            gif.AddFrame(vis);
        }

        //time iteration
        for (int time = 0; time <= totalSimulationTime; time++) {

            // ADD GIFS, PRINT STATS AND SAVE AVERAGE DATA
            if ((time % averageTimeStep) == 0) {
                if (gif != null) {
                    gif.AddFrame(vis);
                }
                tumorEvolution.UpdateStats(time);
                if (abmParameters.nThreads == 0) {
                    tumorEvolution.PrintStats(time);
                }
                tumorEvolution.SaveAverageData(time);
            }

            // Write cell data
            if ((time % iSCTimeStep) == 0 && (seed <= TotaliSCReplicates)) {
                tumorEvolution.WriteiSCData(time);
            }

            //EXECUTE STEP FUNCTION
            tumorEvolution.Step(time, DivisionDependentPmut, DivisionIndependentPmut);

            //CHECK IF POPULATION IS NOT EXTINCT
            if (tumorEvolution.Pop() == 0) {
                tumorEvolution.areAllDead = true;
                break;
            }

        }
        if (gif != null) {
            gif.Close();
        }

        tumorEvolution.deadCellDataFile.Close();
        tumorEvolution.averagesFile.Close();
        tumorEvolution.cellDataFile.Close();
    }

    //////////////////////        WRITE THE DATA    /////////////////////

    void UpdateStats(int time) {
        int n_cells = 0;
        double pdie_cnt = 0;
        double pdiv_cnt = 0;
        int neutrals_cnt = 0;
        int negative_muts_cnt = 0;
        int positive_muts_cnt = 0;
        double positive_mut_impacts_cnt = 0;
        double negative_mut_impacts_cnt = 0;
        int n_dividing = 0;
        double pspace = 0;

        for (Cell c : this) {
            if (c.IsAlive()) {

                if (c.isDividing) {
                    n_dividing++;
                }

                //GROUP1 AVERAGE DATA
                n_cells++;
                pdie_cnt += c.MeanPdie;
                pdiv_cnt += c.MeanPdiv;
                neutrals_cnt += c.CountNeutralMutations;
                positive_muts_cnt += c.CountPositiveMutations;
                negative_muts_cnt += c.CountNegativeMutations;
                positive_mut_impacts_cnt += c.PresentPositiveMutationImpact;
                negative_mut_impacts_cnt += c.PresentNegativeMutationImpact;
                pspace += c.Pspace;
            }
        }
        //Document counts at every iteration. This is to establish average measures
        TotalCellCount = n_cells;
        TotalPdie = pdie_cnt;
        TotalPdiv = pdiv_cnt;
        TotalNeutralMutationsCount = neutrals_cnt;
        TotalPositiveMutationsCount = positive_muts_cnt;
        TotalNegativeMutationsCount = negative_muts_cnt;
        TotalPositiveImpact = positive_mut_impacts_cnt;
        TotalNegativeImpact = negative_mut_impacts_cnt;
        ActivelyDividingCellsCount = n_dividing;
        TotalFreeSpots = pspace;
    }

    void PrintStats(int time) {
        System.out.format("t: %5d cells: %5d pdiv_avg: %5f pdie_avg: %5f positive: %5d negative: %5d no_dividing: %5d \n",
                time, TotalCellCount, TotalPdiv, TotalPdie, TotalPositiveMutationsCount, TotalNegativeMutationsCount, ActivelyDividingCellsCount);
    }

    // write populations every timestep: average and digital signle-cell data
    void SaveAverageData(int time) {
        if (averagesFile != null && !areAllDead) {
            String str_meow = time + "," + TotalCellCount + "," + TotalPdiv +
                    "," + TotalPdie + "," + TotalPositiveMutationsCount + "," + TotalNeutralMutationsCount +
                    "," + TotalNegativeMutationsCount + "," + TotalPositiveImpact +
                    "," + TotalNegativeImpact + "," + ActivelyDividingCellsCount +
                    "," + TotalFreeSpots + "\n";
            averagesFile.Write(str_meow);
            try {
                averagesFile.Flush();

            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    void WriteiSCData(int time) {
        for (Cell c : this) {
            if (cellDataFile != null) {
                if (c.IsAlive()) {
                    String str = time + "," + c.MeanPdiv + "," + c.MeanPdie +
                            "," + c.AccumulatedMutationImpact + "," + c.CountPositiveMutations +
                            "," + c.CountNegativeMutations + "," + c.CountNeutralMutations +
                            "," + c.LineageID + "," + c.Pspace + "\n";
                    cellDataFile.Write(str);
                }
            }
        }
    }

    void WriteDeadCellData(Cell c, int time, int dead_cell_mod) {
        if (deadCellDataFile != null) {
            if (!c.IsAlive() && (time % dead_cell_mod == 0)) {
                String str = time + "," + c.MeanPdiv + "," + c.MeanPdie +
                        "," + c.AccumulatedMutationImpact + "," + c.CountPositiveMutations +
                        "," + c.CountNegativeMutations + "," + c.CountNeutralMutations +
                        "," + c.LineageID + "," + c.Pspace + "\n";
                deadCellDataFile.Write(str);
            }
        }
    }


}
