import pandas as pd
import numpy as np
from typing import List, Tuple


class Canon:
    """Class representing the canonical form of a node conditional probability distribution"""
    
    J : pd.DataFrame
    "The so-called information matrix - the inverse of the covariance matrix (when it exists!)"
    h : pd.Series
    g : float

    def __init__(self, J : pd.DataFrame, h : pd.Series, g : float) -> None:
        """Create a new factor from the canonical parameters.

        The probability density function p(X) (X can be multiple r.v.s) is given by:

        p(X) = exp(g + h.T @ X + 0.5 * (X.T @ J @ X))
        
        The variables in J are the variables in the factor.
        """
        self.J = J
        self.h = h
        self.g = g

    def __mul__(self, other) -> "Canon":
        """Multiplication of factors. Returns a new factor"""
        newJ = pd.DataFrame.add(self.J, other.J, fill_value=0.0).fillna(0.0)
        newH = pd.Series.add(self.h, other.h, fill_value=0.0).fillna(0.0)
        newG = self.g + other.g
        return Canon(newJ, newH, newG)

    def __truediv__(self, other) -> "Canon":
        """Division of factors. Returns a new factor."""
        # Variable name in self.J and other.J must be the same
        newJ = pd.DataFrame.subtract(self.J, other.J, fill_value=0.0).fillna(0.0)
        # Variable name in self.h and other.h must be the same
        newH = pd.Series.subtract(self.h, other.h, fill_value=0.0).fillna(0.0)
        newG = self.g - other.g
        return Canon(newJ, newH, newG)

    
    def marginalize(self, variablesToMarginalize : List) -> "Canon":
        """Integrates out the variables in 'variablesToMarginalize'. Returns a new factor over the remaining ones."""
        
        # We want to keep these variables
        X = list(filter(lambda x: x not in variablesToMarginalize, self.J.index.values))

        # These are the variables we want to marginalize out.
        Y = variablesToMarginalize

        # Converting to numpy arrays
        XX = self.J.loc[X,X].to_numpy()
        YY = self.J.loc[Y,Y].to_numpy()
        XY = self.J.loc[X,Y].to_numpy()
        YX = self.J.loc[Y,X].to_numpy()
        hX = self.h[X].to_numpy()
        hY = self.h[Y].to_numpy()
        YY_inv = np.linalg.inv(YY)

        # Marginalizing out the Y variables and getting the new canonical parameters.
        J = XX - XY @ YY_inv @ YX
        h = hX - XY @ YY_inv @ hY
        g = self.g + 0.5 * (np.log(np.linalg.det(2 * np.pi*YY_inv)) + (hY.T) @  YY_inv @ hY)
        J = pd.DataFrame(J, index = X, columns = X)
        h = pd.Series(h, index = X)
        
        return Canon(J, h, g)

    def Reduce(self, evidence : dict) -> "Canon":
        """Instantiate variables to specific values, e.g. state = {"X1" : 4000, "X4" : 3920}
        
        These variables will no longer be part of the scope of the factor.

        The possible remaining variables will have their probabilities adjusted according to the evidence.

        This process is not reversible, therefore it returns a copy.
        """
        
        if (not np.any([key in self.scope() for key in evidence.keys()])):
            return self

        # X is a list of variables from J that is not in the evidence.
        X = list(filter(lambda x: x not in evidence.keys(), self.scope()))

        # Y is a list of variables that is in the J matrix and the evidence
        Y = list(filter(lambda x: x in self.scope(), evidence.keys()))
        y = np.array([evidence[y] for y in Y])

        XX = self.J.loc[X,X]
        YY = self.J.loc[Y,Y].to_numpy()
        XY = self.J.loc[X,Y].to_numpy()
        hY = self.h[Y].to_numpy()
        hX = self.h[X].to_numpy()

        # J is reduced to just the XX matrix since we no longer need the Y's since they are in the evidence.
        # The other parameters are reduced as well according to equation 14.6 from PGM p. 611
        J = XX
        h = hX - XY @ y
        g = self.g + hY.T @ y - 0.5 * (y.T @ YY @ y)
        h = pd.Series(h, index = X)

        return Canon(J, h, g)

    def scope(self) -> List[str]:
        """Returns the variables that the factor is over, e.g. p("wealth"|"age") is a factor over "wealth" and "age" """
        return self.J.index.values


    def evaluatePDF(self, state : dict, logSpace : bool = False) -> float:
        """Returns the probability density function evaluated at the given state.
        
        state must be a dictionary of the form {variableName : value}.
        
        In some cases, the probability is so low that underflow will occur,
        for this case you can set logSpace to True, which will return the logarithm of the probability density function.
        This can be useful when comparing probabilities.
        """
        scope = self.scope()
        xVector = np.array([state[y] for y in scope])
        product = -0.5 * xVector.T @ self.J @ xVector + xVector.T @ self.h + self.g

        if logSpace:
            return product
        else:
            return np.exp(product)

    @staticmethod
    def fromLinearGaussianCPD(variableName : str, parentName : str,  meanBias : float, parentFactor : float,  variance : float) -> "Canon":
        """Converts a linear conditional gaussian distribution to a canonical form.
        
        The linear conditional gaussian distribution is given by:
        
        p(X | Y) = N(bias + parentFactor * Y, variance)
        
        e.g., the outcome of Y linearly influences the mean of X."""
        index = [variableName, parentName]
        J = pd.DataFrame([[1/variance, -parentFactor/variance],[-parentFactor/variance, (parentFactor**2)/variance]] , index = index, columns = index)
        h = pd.Series([meanBias/variance, -meanBias*parentFactor/variance], index = index)
        g = -(meanBias*meanBias)/(2*variance) - np.log(np.sqrt(2*np.pi*variance))

        return Canon(J, h, g)


    @staticmethod
    def fromUniVariateGaussian(variableName:str, mean : float, variance : float) -> "Canon":
        """Converts a classic univariate Gaussian (x ~ N(µ, s^2)) to a canonical form"""
        return Canon(pd.DataFrame([1/variance], index=[variableName], columns=[variableName]), pd.Series(mean/variance, index=[variableName]), -0.5*mean*mean/variance - np.log(np.sqrt(2*np.pi*variance)))

    def getMean(self) -> pd.Series:
        """Returns the mean of the factor, a.k.a. the expected value of the distribution"""
        newJ = pd.DataFrame(np.linalg.inv(self.J), columns = self.J.columns, index = self.J.index)
        return newJ.dot(self.h)
    
    def getCovariance(self) -> pd.DataFrame:
        """Returns the covariance matrix of the canonical form"""
        return pd.DataFrame(np.linalg.inv(self.J), columns = self.J.columns, index = self.J.index)


from typing import Set

class Funcs:
    @staticmethod
    def Sum_Product_VE (factors : Set[Canon], order : List[object]) -> Canon:
        """Implementation of the Variable Elimination algorithm.
        
        Given a set of factors, this function returns the joint distribution of the variables not in 'order'.

        Variables will me marginalized out in the order they are given in the list.

        Choosing the order of variables is important, as it will affect the sizes (and thereby computation speed) of intermediate factors, however any order will work.
        """

        for variable in order:
            
            # filter out the factors which contain "variable" in their scope
            factorsWithVariableInScope = \
                set(filter(
                    lambda x: variable in x.scope(), 
                    factors))
            
            if (len(factorsWithVariableInScope) == 0):
                continue

            restOfFactors = factors.difference(factorsWithVariableInScope)

            if len(factorsWithVariableInScope) == 0: 
                continue

            else:
                pass

            # multiplying factors togeather to get the psi-factor
            psi = factorsWithVariableInScope.pop()
            for factor in factorsWithVariableInScope:
                psi = psi * factor

            # integrate out the variable
            tau = psi.marginalize([variable])

            # update our set of factors with tau
            factors = restOfFactors.union([tau])

        # multiply remaining factors together
        phi = factors.pop()
        for factor in factors:
            phi = phi * factor

        return phi


    @staticmethod
    def Sum_Product_VE_conditional(factors : Set[Canon], query : List, evidence : dict, order : List, normInLogSpace : bool = False) -> Tuple[float, Canon]:
        """Implementation of the Variable Elimination algorithm with support for giving evidence to initially reduce factor-sizes."""
        reducedFactors = set()

        for factor in factors:
            reducedFactors.add(factor.Reduce(evidence))
        
        
        # remove the query variable from the elimination process
        order = filter(lambda v: v not in query, order)

        psi = Funcs.Sum_Product_VE(reducedFactors, order)
        # normalization is a canonical form with empty J and h, only g is defined. exp(g) is its value
        

        if not normInLogSpace:
            normalization = np.exp(psi.marginalize(query).g)
        else :
            normalization = psi.marginalize(query).g

        return (normalization, psi)
