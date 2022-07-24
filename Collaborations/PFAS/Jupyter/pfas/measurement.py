import pandas as pd
import numpy as np
from substance_dataset import SubstanceDataset


class Measurement:
    """
    This is a container for a PFAS measurement. It combines substance names,
    limits, and measured values with methods to calculate different measures.

    The substance names and limits are specified in a separate csv file
    (``substances.csv``)

    Measurements below the detection limit are assumed to be negative
    values, and their absolute value should be the detection limit,
    i.e. if a substance with detection limit 0.5 is not detected in the
    sample, the value should be -0.5.

    Attributes
    ----------
    substances : pd.DataFrame
        DataFrame holding the substance data in `substances.csv`.
    data : pd.Series
        Series of measured values with compound names as index.
    """

    def __init__(self, data, substance_names=None):
        """
        Parameters
        ----------
        data : array_like or pandas datatype
            The measured concentrations.  Measurements below the detection
            limit are assumed to be negative values, and their absolute value
            should be the detection limit, i.e. if a substance with detection
            limit 0.5 is not detected in the sample, the value should be -0.5.

            This should be either a list/array of values, or a pandas datatype
            like pandas.DataFrame or pandas.Series.
            In the former case, `substance_names` has to be given, which has to
            be a list or array of the names of the substances corresponding to
            the data.
            In the latter case, it is assumed that the index contains the
            substance names.
        substance_names : array_like or None, optional
            Names of the substances in `data`. Only required if `data` is not a
            pandas datatype.
        """
        self.substances = SubstanceDataset()

        if isinstance(data, pd.DataFrame):
            if data.shape[1] != 1:
                raise ValueError("data has too many columns!")
            self.data = data.iloc[:, 0]
        elif isinstance(data, pd.Series):
            self.data = data
        else:
            if substance_names is None:
                raise ValueError(
                    "If data is not a pandas datatype, "
                    "substance_names is a required argument!"
                )
            self.data = pd.Series(data, index=substance_names)

        # make sure the names are in substances
        valid_names = self.substances.index.values
        for name in self.data.index.values:
            if name not in valid_names:
                raise ValueError(
                    f"Invalid name: {name} in data/substance_names"
                )

    def get_concentration(self, name):
        return self.data[name]

    def get_limit(self, name):
        return self.substances.loc["limit", name]

    def get_vorl_gfs(self, name):
        return self.substances.loc["vorl_gfs", name]

    def quotient_sum(self, limit="limit"):
        """
        Calculates the sum of quotients of measured concentrations and limits.

        Parameters
        ----------
        limit : str, optional
            Which limit to use. Either "limit" (default) or "vorl_gfs".

        Returns
        -------
        sum : float
            Quotient sum
        quotients_contributing : array
            Array of single quotients of substances that contributed to the sum
        names_contributing : list of strings
            List of names of substances that contributed to the quotient sum

        Example
        -------
        >>> import measurement
        >>> substances = SubstanceDataset()
        >>> data = [-0.01, -0.01, 1.00e-02, -0.01, 1.20e-01, 3.40e-01, 3.25e+00, 7.00e-02, 6.00e-02, -0.01, -0.01, -0.01, -0.01, 1.95e+00, -0.01, 1.40e-01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01]
        >>> measurement.Measurement(data, substances.index).quotient_sum()
        (26.368333333333332, array([1.66666667e-03, 1.20000000e+00, 5.66666667e+00, 1.95000000e+01]), array(['PFHxA', 'PFOA', 'PFNA', 'PFOS'], dtype=object))
        """
        limits = self.substances[limit]  # can be either "limit" or "vorl_gfs"
        contributing = (limits != 0) & (self.data >= 0)
        quotients_contributing = (
            self.data[contributing] / limits[contributing]
        ).values
        sum_ = np.sum(quotients_contributing)
        names_contributing = contributing[contributing].index.values
        return sum_, quotients_contributing, names_contributing


def quotient_sum_from_series(s, limits):
    contributing = (limits != 0) & (s >= 0)
    quotient = s[contributing] / limits[contributing]
    return quotient.sum()


def quotient_sum_from_dataframe(df, limit="limit"):
    """
    Calculates the quotient sums for entries in a pandas.DataFrame.

    This assumes that the DataFrame has the single measurements as rows, and
    some of the column names are substance names.

    The limit used for the calculation can either be "limit" or "vorl_gfs"
    """
    sd = SubstanceDataset()
    substances = [name for name in df.columns if name in sd.index]
    sd = sd.loc[substances]
    data = df.loc[:, substances]
    limits = sd[limit]
    return data.apply(lambda s: quotient_sum_from_series(s, limits), axis=1)


if __name__ == '__main__':
    import doctest
    print(doctest.testmod())
