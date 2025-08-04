# Add to top of file:
__version__ = "2.0.0"
__author__ = "Tourism Analytics Team"
__all__ = ['CapLimit', 'MinorMerge', 'unpivot', 'smart_round_by_group', 'column_dist']


import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CapLimit:

    """
        to maintain  the average nights cap 'in our case = 90' per trip without changing the whole total  nights per group

        parameters
        ----------
        
        data : pd.DataFrame
          the dataset we are interested in  redistributing  nights in

        cols : list 
        : the columns we group nights by to maintain group sum the same

        cap : int 
        : the maximum number of nights per trip 

        nights : str
          : name of column representing total nights 

        trips : str
          : column represent total trips


    """
    def __init__(self, data:pd.DataFrame , cols:list =None, cap:int = 90 , nights:str='night_wt_sum' , trips:str = 'new_wt_sum'):
        self.data = data
        self.group_columns = cols if cols is not None else [
        'Year', 'FWmonth', 'Trip_Type', 
        'POV', 'Q_13_O1', 'Q_39N'
    ]
        self.cap = cap
        self.nights =nights
        self.trips=trips
    def apply_cap(self):
            # Step 1 : 
            self._calculate_capped_nights()
            self._calculate_available_nights()  
            self._calculate_grouped_values()
            self._calculate_spend_capped_portion()
            self._handle_impossible_cases()
            self._calculate_new_nights()
            self._validate_caps()

            self.cleaned_data = self.data.drop(columns=['cap_nights', 'capped_nights', 'available_nights', 'group_avail_nights', 
                                                      'group_nights', 'group_trips', 'group_capped_nights', 'group_remaining_nights', 'spend_capped_portion', 'impossible_case'])
            return self.cleaned_data

    def _calculate_capped_nights(self):
        """
        calculating top limit for each row , and total nights applying it
        """
        # calculating nights after limit
        self.data['capped_nights'] = self.data[self.nights].clip(upper=self.cap * self.data[self.trips])
        # calculating limit  
        self.data['cap_nights'] = (self.cap * self.data[self.trips]) * (self.data['Trip_Type'] == 'Tourist Trip')
    def _calculate_available_nights(self):
        """ calculating available nights , 
        which is the number of nights remaining in the subcategory that can be increased to absorb 
        nights in cases that exceed thier limit in the same subgroup
        """
        # simply getting number of nights to reach limit of 90*trips
        mask = (self.data['Trip_Type'] == 'Tourist Trip') & ((self.data['cap_nights'] - self.data[self.nights]) > 0)
        self.data['available_nights'] = (self.data['cap_nights'] - self.data[self.nights]).where(mask, 0)
        # self.data.loc[self.data[self.trips] <10 , 'available_nights'] =0
    def _calculate_grouped_values(self):
        """
        to get available nights , and nights to be absorbed  (over 90*trips) per group
        """
        self.data['group_avail_nights'] = self.data.groupby(self.group_columns)['available_nights'].transform('sum')
        self.data['group_nights'] = self.data.groupby(self.group_columns)[self.nights].transform('sum')
        self.data['group_trips'] = self.data.groupby(self.group_columns)[self.trips].transform('sum')
        self.data['group_capped_nights'] = self.data.groupby(self.group_columns)['capped_nights'].transform('sum')
        self.data['group_remaining_nights'] = self.data['group_nights'] - self.data['group_capped_nights']
    def _calculate_spend_capped_portion(self):
        """ to get the portion of each case of nights to be absorbed 
        """
        self.data.loc[self.data['group_avail_nights'] > 0, 'spend_capped_portion']  = self.data['available_nights'] / self.data['group_avail_nights']
        self.data.loc[self.data['group_avail_nights'] == 0, 'spend_capped_portion'] = self.data[self.nights] / self.data['group_nights']
    def _handle_impossible_cases(self):

        self.data['impossible_case'] = 0
        self.data.loc[(self.data['Trip_Type'] == 'Tourist Trip') & (self.data['group_remaining_nights'] > self.data['group_avail_nights']), 'impossible_case'] = 1
    def _calculate_new_nights(self):
        """ Calculate new nights based on caps and remaining nights"""
        self.data.loc[((self.data['group_remaining_nights'] > 0) & (self.data['impossible_case'] == 0)), 'new_nights'] = self.data['capped_nights'] + (self.data['group_remaining_nights'] * self.data['spend_capped_portion'])
        self.data.loc[~((self.data['group_remaining_nights'] > 0) & (self.data['impossible_case'] == 0)), 'new_nights'] = self.data['capped_nights']
        self.data.loc[(self.data['group_remaining_nights'] > 0) & (self.data['impossible_case'] == 1), 'new_nights'] = self.data['capped_nights'] + (self.data['group_avail_nights'] * self.data['spend_capped_portion'])
    
    def _validate_caps(self):
        """ Validate that caps are respected"""
        # Check for cap violations
        tourist_mask = self.data['Trip_Type'] == 'Tourist Trip'
        cap_violations = (
            self.data.loc[tourist_mask, 'new_nights'] > 
            self.data.loc[tourist_mask, 'cap_nights'] + 1e-10  # Small tolerance for floating point
        )
        
        if cap_violations.any():
            n_violations = cap_violations.sum()
            logger.warning(f"WARNING: {n_violations} cap violations detected!")
            
            # Show the violations
            violation_data = self.data.loc[tourist_mask][cap_violations]
            print("Violation details:")
            print(violation_data[['new_nights', 'cap_nights']].head())
        
        # Summary statistics
        total_original = self.data.groupby(self.group_columns).sum().reset_index()[self.nights]
        total_final = self.data.groupby(self.group_columns).sum().reset_index()['new_nights']
        max_difference = max(abs(total_original - total_final))
        try : 
            assert np.allclose(total_original, total_final, rtol=1e-6), "Group totals not preserved!"
        except AssertionError as e :
            print( e)


        
        
        print(f"Cap Application Summary:")
        print(f"  Original total nights: {total_original.sum():.2f}")
        print(f"  Final total nights: {total_final.sum():.2f}")
        print(f"  Maximum difference per subgroup : {max_difference:.2f}")
        print(f"  Cap violations: {cap_violations.sum() if tourist_mask.any() else 0}")
    
        
class MinorMerge :

    """
    merges minor values in a certain set of rows merging them (adding them) into largest value maintain thier sum per row 

    parameters :
    ------------
    df: pd.DataFrame 
     the dataframe in which i need to handle minor values   
    cols: list 
     the set of columns in which we merge minor values to the max value in row
    
     total_col : str 
      name of column that represents total per row
    
    minimum_limit : int
      the value represents lower bound of values to be trimmed
    
    maximum_limit : int
      the value represents upper bound of values to be trimmed    



    """
    def __init__(self , df:pd.DataFrame , cols:list , total_col:str , minimum_limit =0, maximum_limit=1):
        self.df = df.copy(deep=True)
        self.cols = cols
        self.total_col = total_col
        self.minimum_limit = minimum_limit
        self.maximum_limit = maximum_limit
        self.missing_values , self.missing_trips = self._minor_determine()
    

    
    def _minor_determine(self) :
        missing_values=(self.df[self.cols]>self.minimum_limit) & (self.df[self.cols]<self.maximum_limit)
        missing_trips=(missing_values * self.df[self.cols]).sum(axis=1)
        idx = self.df.loc[self.df[self.total_col] > self.maximum_limit].index
        logger.info(f"total trips less than {self.maximum_limit} is {missing_trips[idx].sum()} ")
        return missing_values , missing_trips
    
    def minor_merge(self):
        max_col = self.df.loc[:, self.cols].idxmax(axis=1) 
        top_col=pd.get_dummies(max_col).reindex(columns=self.cols, fill_value=0)
        repeated_missing_values = pd.DataFrame({f'Col_{i}': self.missing_trips for i in range(len(self.cols))})
        repeated_missing_values.columns = self.cols
        new_cols=(repeated_missing_values ).mul(top_col) + ((1-self.missing_values)*self.df[self.cols])
        self.df[self.cols] = new_cols
        if self._minor_validate() :
            logger.info("Done !!")
        else :
            logger.warning("differences exceeded limit")


        
        return self.df

    def _minor_validate(self):
        max_diff = abs((self.df[self.cols].sum(axis=1) - self.df[self.total_col])).max()
        try:
            assert(max_diff < 1.e-09 )
            logger.info("distributed successfully")
            return True
        except AssertionError as e  :
            logger.warning(f"differences exceeded limit {e}")
            return False
                





def unpivot(df:pd.DataFrame , id_vars:list, value_vars,    normalize=False , var_name='variable', value_name='value'):
    """
    Unpivot a DataFrame from wide to long format.
    
    Parameters:
    - df: DataFrame to unpivot
    - id_vars: Columns to keep as identifiers
    - value_vars: Columns to unpivot
    - var_name: Name for the variable column
    - value_name: Name for the value column
    
    Returns:
    - Unpivoted DataFrame
    """

    df_temp = df.copy()



    if not all(col in df.columns for col in id_vars):
        logger.warning(f"Some id_vars {id_vars} are not in the DataFrame columns")
    if not all(col in df.columns for col in value_vars):
        logger.warning(f"Some value_vars {value_vars} are not in the DataFrame columns")
    if set(id_vars) & set(value_vars):
        overlap = list(set(id_vars) & set(value_vars))
        logger.warning(f"id_vars and value_vars overlap: {overlap} to drop them from values set" )
        value_vars = [col for col in value_vars if col not in overlap]
    
    
    # df_temp = df_temp.loc[(df_temp[value_vars].sum()==0) , : ]
    
    if normalize:
        row_sums = df_temp[value_vars].sum(axis=1)
        
        # Identify and remove zero-sum rows
        zero_sum_mask = row_sums == 0
        df_temp = df_temp[~zero_sum_mask].copy()
        
        # Recalculate sums for remaining rows
        row_sums = row_sums[~zero_sum_mask]
        
        # Vectorized normalization
        df_temp.loc[:, value_vars] = df_temp[value_vars].div(row_sums, axis=0)
    

    return df_temp.melt(id_vars=id_vars, value_vars=value_vars,
                         var_name=var_name,
                           value_name=value_name 
                           , ignore_index=False).reset_index(drop=True)


def smart_round_by_group(df, value_col, group_cols, output_col='final' , decimal_spaces=0):
    df = df.copy()

    def smart_round(group):
    # def fast_round(group):
        values = group[value_col].values
        if decimal_spaces == 0:
            floors = np.floor(values).astype(int)
        else:
            floors = np.round(values, decimal_spaces)
        
        target_sum = int(np.round(values.sum(), decimal_spaces))
        decimals = values - floors
        
        
        diff = int(target_sum - floors.sum())
        
        if diff > 0:
            # Use argpartition for O(n) instead of O(n log n) sorting
            top_indices = np.argpartition(decimals, -diff)[-diff:]
            floors[top_indices] += 1
        group[output_col] = pd.Series(floors, index=group.index, name=output_col)

        # group = group.set_index('_original_index') 
        return group[[output_col]]
        
    result = df.groupby(group_cols, group_keys=False).apply(smart_round)


    #df[output_col] = result[output_col]
    return result[output_col]



def column_dist(df , column_group:list , sum_group:str , decimal_spaces=0):
    if decimal_spaces > 0:
        df[column_group] = df[column_group].apply(lambda x: np.round(x, decimal_spaces))
    else :
        df[column_group]=df[column_group].fillna(0).apply(np.floor)
    difference = df[sum_group] - (df[column_group].sum(axis=1)) 
    max_col = df.loc[:, column_group].idxmax(axis=1) 
    difference_absorbed=pd.get_dummies(max_col).reindex(columns=column_group, fill_value=0).mul(difference, axis=0)
    df.loc[: , column_group]+=(difference_absorbed)
    return df




for handler in logger.handlers:
    if isinstance(handler, logging.FileHandler):
        print(f"Log file location: {handler.baseFilename}")
    elif isinstance(handler, logging.handlers.RotatingFileHandler):
        print(f"Rotating log file location: {handler.baseFilename}")
    elif isinstance(handler, logging.handlers.TimedRotatingFileHandler):
        print(f"Timed rotating log file location: {handler.baseFilename}")