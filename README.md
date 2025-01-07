# AdaptationMutation


Contains files related to the manuscript [The adaptive state determines the impact of mutations on evolving populations (bioarxiv)](doi.org/10.1101/2024.12.11.627972)

## movies_simulations 
Contains .gif files of the ABM simulations for pmut = 0 and pmut = 1; each folder contains 3 independent simulations. 

## movies_fitness_evolution 
Contains .gif files of accumulated single-agent data from multiple simulations evolving over time for p<sub>div</sub> = 0.25 (W<sub>0</sub> = 0 for p<sub>die</sub> = 0.2) and p<sub>div</sub> = 1 (W<sub>0</sub> = 0.75 for p<sub>die</sub> = 0.2). 

## ABMcode 
Contains the source code (Java) for ABM simulations. Critical dependencies: [HAL](https://github.com/MathOnco/HAL) and [Jcommander](https://jcommander.org/) 

## Analysis
Contains the code (Python) to analyze the simulations (.csv) generated from the code in ABMcode (UNDER CONSTRUCTION)
