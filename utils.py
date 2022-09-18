import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
import os
import ipywidgets as widgets

def prepare_dataset(dataset_name='iris', test_size=0.25):
    if dataset_name=='iris':
        data_df = pd.read_csv('data/iris.csv')
        data_df.dropna(0, inplace=True)
        encoder_species = LabelEncoder()
        X = data_df.iloc[:,:-1].values
        y = np.ravel(encoder_species.fit_transform(data_df['species']))

    elif dataset_name=='penguins':
        data_df = pd.read_csv('data/penguins.csv')
        data_df.dropna(0, inplace=True)
        encoder_island = LabelEncoder()
        encoder_sex = LabelEncoder()
        encoder_species = LabelEncoder()
        data_df['island'] = np.ravel(encoder_island.fit_transform(data_df['island']))
        data_df['sex'] = np.ravel(encoder_sex.fit_transform(data_df['sex']))
        X = data_df.iloc[:,1:].values
        y = np.ravel(encoder_species.fit_transform(data_df['species']))

    else:
        print('Please prepare dataset yourself.')
        X=None; y = None
    return X, y

def scatter_plot(current_alg, acc, dataset_name, classification=True):
    if classification:
        columns=['Algorithm','Test accuracy','Current']
        filenames =['class_knn.csv', 'class_svm.csv', 'class_rf.csv']
        algorithms = ['KNN', 'SVM', 'Random Forest']
        threshold=0.7
        lookup_column='Mean CV test acc'
    else:
        columns=['Algorithm','Test neg-MSE','Current']
        filenames =['reg_rf.csv', 'reg_lin.csv', 'reg_log.csv']
        algorithms = ['Random Forest', 'Linear Regression', 'Logistic Regression']
        threshold=-10
        lookup_column='Mean CV test neg-MSE'

    too_few_trained =[]
    total_df = pd.DataFrame(columns=columns)
    for filename, algorithm in zip(filenames, algorithms):
        try:
            outdir = './trained_models/'+ dataset_name
            fullname = os.path.join(outdir, filename)
            alg_df = pd.read_csv(fullname ,sep=';')
            hp_columns = list(alg_df.columns)
            for column_name in ['Mean CV test acc', 'Mean CV train acc', 'Mean CV test neg-MSE', 'Mean CV train neg-MSE', 'Fitting time', 'random_state']:
                try:
                    hp_columns.remove(column_name)
                except:
                    hp_columns=hp_columns
            alg_temp = pd.DataFrame(columns=columns)
            alg_temp['Algorithm'] = [algorithm]*len(alg_df)
            if len(alg_df) <= 3:
                too_few_trained.append(algorithm)
            alg_temp[columns[1]] = alg_df[lookup_column]
            alg_temp['hp_dict'] = alg_df[hp_columns].to_dict(orient='record')
            alg_temp['hp_dict'] = alg_temp['hp_dict'].astype(str)
            alg_temp['hp_dict'] = alg_temp['hp_dict'].str.replace(',','<br>')
            alg_temp['Current'] = np.where(((alg_temp[columns[1]] == round(acc, 5)) & (alg_temp['Algorithm'] == current_alg)), 0.3, 0.02)
            total_df = pd.concat([total_df, alg_temp], ignore_index=True)
        except:
            too_few_trained.append(algorithm)
    if len(total_df.index) >0:
        total_df_new = total_df[((total_df[columns[1]]>=threshold) | (total_df['Current']==True))]
        if len(total_df_new)<5:
            total_df_new=total_df
        
        size = total_df_new['Current']
        total_df_new.drop(columns='Current', inplace=True)

        fig = px.scatter(total_df, x=columns[1], y='Algorithm', color='Algorithm', size=size, hover_name='Algorithm', hover_data=['hp_dict'])
        xaxis_title = '{} (>{})'.format(columns[1], threshold)
        fig.update_layout(xaxis_title=xaxis_title)
        fig_widget = go.FigureWidget(fig)
        if len(too_few_trained)>0:
            too_few_trained_str = ', '.join(too_few_trained)
            return [fig_widget] + [widgets.HTML('Too few '+ too_few_trained_str + ' models have been trained. Please try them out for betther overview.')]
        else:
            return [fig_widget]
    else:
        return [widgets.HTML('No models have been trained. Please train some models.')]


def scatter_plot_overview(dataset_name, type, first, second, third):
    if type == 'Classification':
        columns=['Algorithm','Test accuracy', 'Train accuracy', 'Fitting time']
        filenames =['class_knn.csv', 'class_svm.csv', 'class_rf.csv']
        algorithms = ['KNN', 'SVM', 'Random Forest']
        lookup_column1='Mean CV test acc'
        lookup_column2='Mean CV train acc'
    else:
        columns=['Algorithm','Test neg-MSE', 'Train neg-MSE', 'Fitting time']
        filenames =['reg_rf.csv', 'reg_lin.csv', 'reg_log.csv']
        algorithms = ['Random Forest', 'Linear Regression', 'Logistic Regression']
        lookup_column1='Mean CV test neg-MSE'
        lookup_column2='Mean CV train neg-MSE'

    if first==1 & second==1:
        x=columns[1]; y=columns[2]
    elif first==1 & third==1:
        x=columns[1]; y=columns[3]
    elif second==1 & third==1:
        x=columns[2]; y=columns[3]

    total_df = pd.DataFrame(columns=columns)
    too_few_trained = []
    for filename, algorithm in zip(filenames, algorithms):
        try:
            outdir = './trained_models/'+ dataset_name
            fullname = os.path.join(outdir, filename)
            alg_df = pd.read_csv(fullname ,sep=';')
            hp_columns = list(alg_df.columns)
            for column_name in ['Mean CV test acc', 'Mean CV train acc', 'Mean CV test neg-MSE', 'Mean CV train neg-MSE', 'Fitting time', 'random_state']:
                try:
                    hp_columns.remove(column_name)
                except:
                    hp_columns=hp_columns

            alg_temp = pd.DataFrame(columns=columns)
            alg_temp['Algorithm'] = [algorithm]*len(alg_df)
            if len(alg_df) <= 3:
                too_few_trained.append(algorithm)
            alg_temp[columns[1]] = alg_df[lookup_column1]
            alg_temp[columns[2]] = alg_df[lookup_column2]
            alg_temp[columns[3]] = alg_df['Mean CV fit time']
            alg_temp['hp_dict'] = alg_df[hp_columns].to_dict(orient='record')
            alg_temp['hp_dict'] = alg_temp['hp_dict'].astype(str)
            alg_temp['hp_dict'] = alg_temp['hp_dict'].str.replace(',','<br>')
            total_df = pd.concat([total_df, alg_temp], ignore_index=True)
        except:
            too_few_trained.append(algorithm)
    
    if len(total_df.index) > 0:

        fig = px.scatter(total_df, x=x, y=y, color='Algorithm', hover_name='Algorithm', hover_data=['hp_dict'])
        fig_widget = go.FigureWidget(fig)
        if len(too_few_trained)>0:
            too_few_trained_str = ', '.join(too_few_trained)
            return [fig_widget] + [widgets.HTML('Too few '+ too_few_trained_str + ' models have been trained. Please try them out for betther overview.')]
        else:
            return [fig_widget]
    else:
        return [widgets.HTML('No models have been trained. Please train some models.')]

def feature_importance(dataset_name, alg, alg_type, n=5):
    if alg_type=='Classification':
        alg_pairs = {'knn': 'class_knn.csv',
                    'SVM': 'class_svm.csv',
                    'Random Forest': 'class_rf.csv'}
    else:
        alg_pairs = {'Linear Regression': 'reg_lin.csv',
                    'Logistic Regression': 'reg_log.csv',
                    'Random Forest': 'reg_rf.csv'}
    
    try:
        outdir = './trained_models/'+ dataset_name
        fullname = os.path.join(outdir, alg_pairs[alg])
        alg_df = pd.read_csv(fullname ,sep=';')
        alg_df.dropna(inplace=True)
        if alg_type=='Classification':
            alg_df.dropna(subset=['Mean CV test acc'], inplace=True)
            y = alg_df['Mean CV test acc']
        else:
            alg_df.dropna(subset= ['Mean CV test neg-MSE'], inplace = True)
            y = alg_df['Mean CV test neg-MSE']
        columns_to_drop = list(set(alg_df.columns).intersection(["random_state"]))
        if len(columns_to_drop)>0:
            alg_df.drop(columns=columns_to_drop, inplace=True)    
        alg_df_new = alg_df[alg_df.select_dtypes(exclude=[object]).columns]
        object_columns = alg_df.select_dtypes(include=[object]).columns
        numeric_columns = alg_df.select_dtypes(exclude=[object]).columns
        for column in object_columns:
            column_str = column+'_str'
            column_num = column+'_num'
            alg_df_new[column_str] = alg_df[column].str.extract('(\D+)') 
            alg_df_new[column_num] = alg_df[column].str.extract('(\d+)').apply(pd.to_numeric, errors='coerce')
        columns=alg_df_new.columns.to_list()
        if alg_type=='Classification':
            columns = [e for e in columns if e not in ('Mean CV train acc', 'Mean CV test acc', 'Mean CV fit time')]
        else:
            columns = [e for e in columns if e not in ('Mean CV train neg-MSE', 'Mean CV test neg-MSE', 'Mean CV fit time')]
        alg_df_new.dropna(axis=1, how='all', inplace=True)

        for column in alg_df_new.select_dtypes(include=[object]).columns:
            replacement_dict = {}
            unique_values = alg_df_new[column].unique()
            unique_values = [x for x in unique_values if type(x)==str]
            for i in range(len(unique_values)):
                replacement_dict[unique_values[i]] = i+1
            alg_df_new[column]=alg_df_new[column].replace(replacement_dict).apply(pd.to_numeric, errors='coerce')
        alg_df_new.fillna(-1, inplace=True)

        columns=alg_df_new.columns.to_list()
        if alg_type=='Classification':
            columns = [e for e in columns if e not in ('Mean CV train acc', 'Mean CV test acc', 'Mean CV fit time')]
        else:
            columns = [e for e in columns if e not in ('Mean CV train neg-MSE', 'Mean CV test neg-MSE', 'Mean CV fit time')]

        X = alg_df_new[columns]
        from sklearn.ensemble import RandomForestRegressor
        regr = RandomForestRegressor()
        regr.fit(X, y)
        feature_importance=regr.feature_importances_
        res = dict(zip(columns, feature_importance))
        for column in object_columns:
            res_short = {key:val for key, val in res.items() 
                        if key.startswith(column)}
            res[column] = sum(res_short.values())
            for key in res_short.keys():
                del res[key]

        res = dict(sorted(res.items(), key=lambda item: item[1], reverse=True))
        n=min(n, len(res))
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        rows=n
        cols=3
        if alg_type=='Classification':
            list_of_metrics = ['Mean CV train acc', 'Mean CV test acc', 'Mean CV fit time'] 
        else:
            list_of_metrics = ['Mean CV train neg-MSE', 'Mean CV test neg-MSE', 'Mean CV fit time']
        fig = make_subplots(rows=rows, cols=cols, start_cell="top-left", column_titles=list_of_metrics)
        for column in range(1, cols+1):
            for row in range(1, rows+1):
                fig.add_trace(go.Scatter(x=alg_df[list_of_metrics[column-1]], y=alg_df[list(res.keys())[row-1]], mode='markers'), 
                            row=row, col=column)
                fig.update_yaxes(row=row, col=column, tickangle=45)
                if column == 1:
                    name = list(res.keys())[row-1]+'<br>({}%)'.format(round(list(res.values())[row-1]*100, 2))
                    fig.update_yaxes(title_text=name, row=row, col=column)

        fig.update_layout(showlegend=False, height=900, width=900)
        fig_widget = go.FigureWidget(fig)

        few_values_list = []
        for column in numeric_columns:
            if alg_df[column].nunique() <= 3:
                few_values_list.append(column)
        if len(few_values_list)>3:
            for m in list_of_metrics:
                if m in few_values_list:
                    few_values_list.remove(m)
            few_values_str = ', '.join(few_values_list)
            comment = widgets.HTML('Too few different values for these numeric hyperparameters: {}. Please train more models using different values for better overview.'.format(few_values_str))
            return [fig_widget] + [comment] 
        else:
            return [fig_widget]
        
    
    except:
        return [widgets.HTML('<No {} models have been trained. Please train before plotting.'.format(alg))]


def parallel_coordinates(dataset_name, alg, alg_type):
    if alg_type=='Classification':
        alg_pairs = {'knn': 'class_knn.csv',
                    'SVM': 'class_svm.csv',
                    'Random Forest': 'class_rf.csv'}
    else:
        alg_pairs = {'Linear Regression': 'reg_lin.csv',
                    'Logistic Regression': 'reg_log.csv',
                    'Random Forest': 'reg_rf.csv'}

    try:
        outdir = './trained_models/'+ dataset_name
        fullname = os.path.join(outdir, alg_pairs[alg])
        alg_df = pd.read_csv(fullname ,sep=';')
        if alg_type == "Classification":
            alg_df.dropna(subset=['Mean CV test acc'], inplace=True)
        else:
            alg_df.dropna(subset=['Mean CV test neg-MSE'], inplace = True)

        if "random_state" in list(alg_df.columns):
            alg_df.drop(columns='random_state', inplace=True)
    
        object_columns = alg_df.select_dtypes('object').columns.to_list()
        not_object_columns = alg_df.select_dtypes(exclude=[object]).columns

        object_list = []
        for column in object_columns:
            unique_values = alg_df[column].unique()
            dfg = pd.DataFrame({column:unique_values})
            new_column = column+"_new"
            dfg[new_column] = dfg.index
            alg_df = pd.merge(alg_df, dfg, on = column, how='left')
            object_dict = dict(range=[0,alg_df[new_column].max()],
                            tickvals = dfg[new_column], ticktext = dfg[column],
                            label=column.replace('_', ' ').title(), values=alg_df[new_column])
            object_list.append(object_dict)

        not_object_list = []
        for column in not_object_columns:
            if column == 'Mean CV test acc' or column == 'Mean CV test neg-MSE' or 'Mean CV train acc' or column == 'Mean CV train neg-MSE' or column == 'Mean CV fit time':
                alg_df[column] = alg_df[column].apply(lambda x: round(x, 2))
            if column == 'Mean CV test acc' or column == 'Mean CV test neg-MSE':
                not_objecct_dict = dict(range=[alg_df[column].min(),alg_df[column].max()],
                    tickvals = alg_df[column], label=column.replace('_', ' ').title(), values=alg_df[column],
                                    constraintrange = [alg_df[column].min(),(alg_df[column].max()-alg_df[column].min())/3])
                not_object_list.append(not_objecct_dict)

            else:
                not_objecct_dict = dict(range=[alg_df[column].min(),alg_df[column].max()],
                    tickvals = alg_df[column], label=column.replace('_', ' ').title(), values=alg_df[column])
                not_object_list.append(not_objecct_dict)
        import plotly.express as px
        coordinates_list = object_list + not_object_list 
        fig = go.Figure(data=go.Parcoords(line = dict(color = alg_df[not_object_columns[-2]], colorscale = 'bluered', showscale = True), 
                                            dimensions=coordinates_list, labelangle=-30))

        fig_widget = go.FigureWidget(fig)

        few_values_list = []
        for column in not_object_columns:
            if alg_df[column].nunique() <= 3:
                few_values_list.append(column)
        if len(few_values_list)>3:
            if alg_type == "Classification":
                list_of_metrics = ['Mean CV train acc', 'Mean CV test acc', 'Mean CV fit time'] 
            else:
                list_of_metrics = ['Mean CV train neg-MSE', 'Mean CV test neg-MSE', 'Mean CV fit time']
            for m in list_of_metrics:
                if m in few_values_list:
                    few_values_list.remove(m)
            few_values_str = ', '.join(few_values_list)
            comment = widgets.HTML('Too few models of {} have been trained. Please train more models for better overview.'.format(few_values_str))
            return [fig_widget] + [comment]
        else:
            return [fig_widget]
    except:
        return [widgets.HTML('<No {} models have been trained. Please train before plotting.'.format(alg))]
