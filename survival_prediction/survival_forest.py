import pandas as pd
import numpy  as np
import argparse
import joblib
from   sklearn.cluster      import FeatureAgglomeration
from   sklearn.pipeline     import Pipeline
from   sksurv.ensemble      import RandomSurvivalForest
from   lifelines.statistics import logrank_test

class Survival(object):
    
    def __init__(self, args):
        self.features_file        = args.features_file
        self.path_to_save_results = args.path_to_save_results
        self.num_estimators_rsf   = args.num_estimators_rsf
        
        # Set up the colors for drawing the Kaplan Meier Curves
        self.mysand  = '#BCA18D'
        self.mygreen = '#118C8B'
        self.myred   = '#F2746B'
        self.Q1color = '#3BCA2B'
        self.Q2color = '#2893CC'
        self.Q3color = '#6B4B3E'
        self.Q4color = '#637074'
    
    def read_file(self):
        features_data = pd.read_csv(self.features_file)
        return features_data[0:100]
    
    
    def is_even(self, patient_id):
        even = (patient_id % 2 == 0)
        return even
    
    
    def get_train_test_data(self):
        # Reading features file and splitting data into train and test
        features_data = self.read_file()
        list_idx = features_data.PatientID.apply(self.is_even)
        data_train, data_test = features_data[list_idx], features_data[~list_idx]
        print('Number of train samples: ', len(data_train), data_train.shape[0])
        print( 'Number of test samples: ', len(data_test), data_test.shape[0])
        return data_train, data_test


    def train_random_survival_forest(self, data_train):
        data_train['Event'] = data_train['Event'].astype(int)
        
        rsf = RandomSurvivalForest(
            n_estimators=self.num_estimators_rsf, random_state=42, n_jobs=-1
        )
        
        # Assign x and y for the
        x_cols = [c for c in data_train.columns if c.startswith('feature') or c.startswith('Difference')]
        y_cols = ['Event', 'DaysOfSurvival']

        # Selecting the right columns and converting to a numpy record array
        data_train_y          = data_train[y_cols]
        data_train_y['Event'] = data_train['Event'] == 1
        data_train_y          = data_train_y.to_records(index=False)
        
        # Train random survival forest
        rsf_model = rsf.fit(data_train[x_cols], data_train_y[y_cols])
        joblib.dump(rsf_model, self.path_to_save_results + "rfs_model.joblib")
        
        
    def load_rsf_model(self):
        rsf_model = joblib.load(self.path_to_save_results + "rfs_model.joblib")
        return rsf_model


    def test_rsf_model(self, data_test, rsf_model):
        x_cols               = [c for c in data_test.columns if c.startswith('feature') or c.startswith('Difference')]
        y_cols               = ['Event', 'DaysOfSurvival']
        data_test['Event']   = data_test['Event'].astype(int)
        data_test_y          = data_test[y_cols]
        data_test_y['Event'] = data_test_y['Event'] == 1
        data_test_y          = data_test_y.to_records(index=False)
        out_result           = rsf_model.score(data_test[x_cols], data_test_y[y_cols])
        return out_result 
        
    
    def test_rsf_model_with_bootstrap(self, data_test, rsf_model, n_repetitions):
        out_results = []
        for i in range(n_repetitions):
            df_sample = data_test.sample(n=data_test.shape[0], replace=True, random_state=i)
            out_results.append(self.test_rsf_model(df_sample, rsf_model))
        return np.median(out_results), [np.percentile(out_results, 2.5), np.percentile(out_results, 97.5)]

    
    def get_p_value(self, df_test_logrank, scores, cutoff):
        p_value = logrank_test(
            df_test_logrank['DaysOfSurvival'][scores<cutoff], 
            df_test_logrank['DaysOfSurvival'][scores>=cutoff],
            df_test_logrank['Event'][scores<cutoff]==1,
            df_test_logrank['Event'][scores>=cutoff]==1
        ).p_value
        return p_value


    def logrank_test(self, data_test, rsf_model):
        data_test_logrank = data_test
        x_cols            = [c for c in data_test.columns if c.startswith('feature') or c.startswith('Difference')]
        scores            = rsf_model.predict(data_test_logrank[x_cols])
        data_test_logrank['score'] = scores
        
        # Cutoffs
        cutoff_1 = np.percentile(scores, 50)
        #cutoff_3 = [0., np.percentile(scores, 33), np.percentile(scores, 66), np.percentile(scores, 100)]
        #cutoff_4 = [0., np.percentile(scores, 25), np.percentile(scores, 50), np.percentile(scores, 75), np.percentile(scores, 100)]

        p_value_1 = self.get_p_value(data_test_logrank, scores, cutoff_1)
        #p_value_3 = self.get_p_value(data_test_logrank, scores, cutoff_3)
        #p_value_4 = self.get_p_value(data_test_logrank, scores, cutoff_4)

        print('General, p-values: ', p_value_1)#, ' ', p_value_3, ' ',  p_value_4)
        
    
    '''def graph_layout(x_label, y_label, title, x_lim=None, y_lim=None, x_step=None, y_step=None, legend=True):
        ax.spines[['right', 'top']].set_visible(False)
        ax.spines[['left', 'bottom']].set_linewidth(4)
        ax.spines[['left', 'bottom']].set_color('black')
        ax.xaxis.set_tick_params(width=4)
        ax.yaxis.set_tick_params(width=4)
        ax.grid(False)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        plt.xticks(fontsize= 24, fontweight='bold')
        plt.yticks(fontsize= 24, fontweight='bold')
        if x_step != None:
            plt.xticks(np.arange(min(x_lim), max(x_lim), x_step))
        if y_step != None:
            plt.yticks(np.arange(min(y_lim), max(y_lim), y_step))

        if legend == True:
            plt.legend(frameon=False,
                    loc='lower left',
                    fontsize = 24)

        plt.xlabel(x_label, fontsize = 32, fontweight='bold')
        plt.ylabel(y_label, fontsize = 32, fontweight='bold')
        plt.title(title, fontsize = 36, fontweight='bold', pad = 20)'''
        
def main(args):
    # Training: Random survival forest
    survival              = Survival(args)
    data_train, data_test = survival.get_train_test_data()
    #survival.train_random_survival_forest(data_train)
    
    # Testing: Performance assessment via C-index
    rsf_model  = survival.load_rsf_model()
    out_result = survival.test_rsf_model(data_test, rsf_model)
    out_result_bootstrap  = survival.test_rsf_model_with_bootstrap(data_test, rsf_model, 100)
    print('Test random survival forest score: ', out_result)
    print('General, c-index: ', out_result_bootstrap)
    
    # Statistical Evaluation: Logrank test
    # Several defined cutoffs to measure the model performance 
    # regarding the different risk-groups 
    logrank_test(data_test, rsf_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing of YADS and patient files to create pairs')
    
    # Variables to preprocess the NKI files: YADS file and patients file
    parser.add_argument('--features_file',         default='/projects/disentanglement_methods/files_nki/infoA/abdomen/features/2.features_pam_including_difference_between_dates_of_treatment.csv', type=str, help='features file path')
    parser.add_argument('--path_to_save_results',  default='/projects/disentanglement_methods/files_nki/infoA/abdomen/survival_results/', type=str, help='Path to save the preprocessed file of YADS')
    parser.add_argument('--num_estimators_rsf',    default=1000, type=int, help='Number of estimators for the random survival forest')
    args = parser.parse_args()
    
    main(args)