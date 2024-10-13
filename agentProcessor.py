import numpy as np
import pandas as pd


class AgentProcessor:
    def __init__(
        self, size, numAgents, AgentFindNeighbours, PostProcessData, CalculateVector
    ):
        initialData = {
            "x": np.around(np.random.rand(numAgents) * size, 2),
            "y": np.around(np.random.rand(numAgents) * size, 2),
            "N_x": [np.nan] * numAgents,
            "N_y": [np.nan] * numAgents,
            "E_x": [np.nan] * numAgents,
            "E_y": [np.nan] * numAgents,
            "W_x": [np.nan] * numAgents,
            "W_y": [np.nan] * numAgents,
            "S_x": [np.nan] * numAgents,
            "S_y": [np.nan] * numAgents,
            "dx": [0.0] * numAgents,
            "dy": [0.0] * numAgents,
        }
        self.df = pd.DataFrame(data=initialData)
        self.AgentFindNeighbours = AgentFindNeighbours
        self.PostProcessData = PostProcessData
        self.CalculateVector = CalculateVector
        self.numAgents = numAgents

    def step(self):
        self.moveAgents()
        dfNeighbour = self.findNeighbours()
        dfVector = self.CalculateVector(dfNeighbour)
        self.df = pd.concat([self.df[["x", "y"]], dfNeighbour, dfVector], axis=1).round(
            2
        )
        return self.df

    def interpolatedStep(self, steps):
        dfs = []
        for i in range(steps - 1):
            self.moveAgents(steps)
            dfs.append(self.df.copy())
        self.moveAgents(steps)
        dfNeighbour = self.findNeighbours()
        dfVector = self.CalculateVector(dfNeighbour)
        self.df = pd.concat([self.df[["x", "y"]], dfNeighbour, dfVector], axis=1).round(
            2
        )
        dfs.append(self.df)
        return dfs

    def moveAgents(self, steps=1):
        xStep = self.df["dx"] / steps
        yStep = self.df["dy"] / steps
        self.df.x += xStep
        self.df.y += yStep

    def findNeighbours(self):
        neighbourData = []
        for agentIndex in range(self.numAgents):
            output = {"agent": agentIndex}
            for neighbourIndex in range(self.numAgents):
                output = self.AgentFindNeighbours(
                    self.df, agentIndex, neighbourIndex, output
                )
            neighbourData.append(output)
        #
        processedNeighbourData = []
        for d in neighbourData:
            processedNeighbourData.append(self.PostProcessData(d))
        dfNeighbour = pd.DataFrame(data=processedNeighbourData)
        dfNeighbour = dfNeighbour.set_index("agent")
        return dfNeighbour
