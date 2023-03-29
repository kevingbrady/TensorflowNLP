import pandas as pd


class DataFrameManager:

    sp800_53 = None
    catalog = None
    csf = None

    def __init__(self, ArgumentManager):

        self.sp800_53 = pd.read_excel(ArgumentManager.sp800_53)
        self.catalog = pd.read_excel(ArgumentManager.catalog, sheet_name="(new)Non-Tech Manufactures")
        self.csf = pd.read_excel(ArgumentManager.csf)

        self.sp800_53.drop(0, inplace=True)
        self.catalog.drop(0, inplace=True)

        self.catalog.columns = ["NON-TECHNICAL", "Sub-Capability", "Element/Action", "Sub-Element/Sub-Action", "800-53",
                           "CSF",
                           "Comment", "Filter by X for comments only", "NIST Reviewer", 'Unnamed: 10']

        self.sp800_53.fillna("", inplace=True)
        self.catalog.fillna(method='ffill', inplace=True)
        self.csf.fillna(method='ffill', inplace=True)

        self.csf.drop(columns=['Informative References'], inplace=True)
        self.csf.drop_duplicates(subset=['Category', 'Subcategory'], inplace=True)

        self.csf['Control Identifier'] = self.csf['Subcategory'].apply(lambda r: r[0:7])

        # Filter out sections with no controls
        self.catalog = self.catalog[self.catalog['Element/Action'].str.contains("^[a-z]\.") == False]

        self.catalog.reset_index(inplace=True)
        self.sp800_53.reset_index(inplace=True)
        self.csf.reset_index(inplace=True)