package Evolvability.Winter2023;

//Simplify the experiments - pre-define parameters controlling mutations
class InvalidCommandLineException extends Exception {
    public InvalidCommandLineException(String string) {
        super(string);
    }
}

// Simplify the experiments - pre-define parameters controlling mutations
enum MutationRun {

    // Define mutation rates used for the experiments and the associated parameters.
    // NOTE: Do not change the below unless you really have to. Doing so will
    // invalidate the 'golden' dataset, so before committing you will have to
    // update the dataset (see run-golden.sh for instructions).
    // This should only be done as a last resort, as it will make it harder to
    // verify
    // correctness between past and future versions.
    MUTATION_RATE_1(1, 1, 25, 1, "both_1"),
    MUTATION_RATE_01(1e-1, 1e-1, 75, 1, "both_1e-1"),
    MUTATION_RATE_02(1e-2, 1e-2, 150, 1, "both_1e-2"),
    MUTATION_RATE_03(1e-3, 1e-3, 250, 1, "both_1e-3"),
    MUTATION_RATE_04(1e-4, 1e-4, 250, 1, "both_1e-4"),
    MUTATION_RATE_0(0, 0, 250, 1, "both_0");

    private double DivisionIndependentPMut;
    private double DivisionDependentPMut;
    private int TotalAverageReplicates;
    private int TotaliSCReplicates;
    private final String MutationNameDefinition;

    MutationRun(double DivisionIndependentPMut, double DivisionDependentPMut,
            int TotalAverageReplicates, int TotaliSCReplicates, String MutationNameDefinition) {

        this.DivisionIndependentPMut = DivisionIndependentPMut;
        this.DivisionDependentPMut = DivisionDependentPMut;
        this.TotalAverageReplicates = TotalAverageReplicates;
        this.TotaliSCReplicates = TotaliSCReplicates;
        this.MutationNameDefinition = MutationNameDefinition;
    }

    /*
     * Updates the MutationRun object from a parameter string. The string must be of
     * format
     * "<parameter>=<value>[,<parameter>=<value>]"
     */
    public void UpdateFromParameterString(String parm) throws InvalidCommandLineException {
        // TODO: This updates the "global instance" of the object. That's probably fine for our
        //       usecase, but generally speaking we should probably create a new object here.
        if (parm.length() == 0) {
            return;
        }
        String[] chunks = parm.split(";");
        for (String chunk : chunks) {
            String[] keyvaluepair = chunk.split("=");
            if (keyvaluepair.length != 2) {
                throw new InvalidCommandLineException("In " + chunk + ": key-value pair has multiple '=' or none.");
            }
            // TODO: Verify that no key is duplicated
            try {
                switch (keyvaluepair[0]) {
                    case "DivisionIndependentPMut": {
                        this.DivisionIndependentPMut = Double.parseDouble(keyvaluepair[1]);
                        break;
                    }
                    case "DivisionDependentPMut": {
                        this.DivisionDependentPMut = Double.parseDouble(keyvaluepair[1]);
                        break;
                    }
                    case "TotalAverageReplicates": {
                        this.TotalAverageReplicates = Integer.parseInt(keyvaluepair[1]);
                        break;
                    }
                    case "TotaliSCReplicates": {
                        this.TotaliSCReplicates = Integer.parseInt(keyvaluepair[1]);
                        break;
                    }
                    default:
                        throw new InvalidCommandLineException(
                                "Invalid variable to update: " + keyvaluepair[0] + " (in chunk: " + chunk + ")");
                }
            } catch (NumberFormatException ex) {
                throw new InvalidCommandLineException("Unable to parse number in chunk: " + chunk);
            }
        }
    }

    public double DivisionIndependentPMut() {
        return DivisionIndependentPMut;
    }

    public double DivisionDependentPMut() {
        return DivisionDependentPMut;
    }

    public int TotalAverageReplicates() {
        return TotalAverageReplicates;
    }

    public int TotaliSCReplicates() {
        return TotaliSCReplicates;
    }

    public String MutationNameDefinition() {
        return MutationNameDefinition;
    }

    static MutationRun GetByName(String name) {
        for (MutationRun m : MutationRun.values()) {
            if (m.MutationNameDefinition().equals(name)) {
                return m;
            }
        }
        return null;
    }
}
