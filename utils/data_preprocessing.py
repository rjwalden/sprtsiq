import pandas as pd
import numpy as np


# feature
def feature_preprocessor(df_raw, ls_outmatch_features_contin, n_rolling):
    '''
    Descriptions
    '''
    df_lookback_tmp = pd.DataFrame(columns=df_raw.columns)
    
    for team in df_raw.Team.unique():
        df_target_team = df_raw[df_raw.Team == team]
        df_target_team = df_target_team.sort_values(by=['Date']).shift(1)[1:]

        for col in ls_outmatch_features_contin:
            df_target_team[col] = df_target_team[col].rolling(n_rolling).median()
            
        df_target_team['PassingYardsXPassingCompletions'] = df_target_team['PassingYards'] * df_target_team['PassingCompletions']
        df_target_team['PassingYardsXFirstDownsByPassing'] = df_target_team['PassingYards'] * df_target_team['FirstDownsByPassing']
        # df_target_team['PC1_allnegVSPointSpread'] = (df_target_team[['RedZoneAttempts', 'Points Scored', 'OffensiveTouchdowns',             'RedZoneConversions', 'OffensiveYards', 'FirstDowns', 'PassingTouchdowns', 'Touchdowns', 'PasserRating']].prod(axis=1)) / df_target_team['PointSpread']
        df_target_team['PC2_ProductOfAllLoadings'] = df_target_team[['RedZoneConversionsAllowed', 'YardsAllowed', 'TotalScore', 'RedZoneAttemptsAgainst', 'PassingYardsAllowed', 'TouchdownsAllowed', 'Points Allowed', 'FirstDownsAllowed', 'PasserRatingAllowed', 'OverUnder']].prod(axis=1)
        df_target_team['PC4_ProductOfAllLoadings'] = df_target_team[['PassingYards', 'PassingCompletions', 'FirstDownsByPassing']].prod(axis=1)
        df_target_team['PC2_YardsAllowedXPointsAllowedXFirstDownsAllowed'] = df_target_team[['YardsAllowed', 'Points Allowed', 'FirstDownsAllowed']].prod(axis=1)
        df_target_team['PC5_RushingAttempsXTimePossessionVSTimeDefense'] = df_target_team[['RushingAttempts', 'TimeOfPossessionInSeconds']].prod(axis=1) / df_target_team['TimeOnDefenseInSeconds']
            
        df_target_team = df_target_team[(n_rolling-1):]
        df_lookback_tmp = pd.concat([df_lookback_tmp, df_target_team])
            
    return df_lookback_tmp

