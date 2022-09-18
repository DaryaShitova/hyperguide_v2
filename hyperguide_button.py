from ipywidgets import Layout, Button, Box, VBox
from IPython.display import display
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import os
import sys
pd.options.mode.chained_assignment = None 
import os.path
import seaborn as sns

import ipywidgets as widgets
from ipywidgets import interactive
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from utils import scatter_plot, scatter_plot_overview, feature_importance, parallel_coordinates

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


class Hyper_Parameter_Guide(widgets.DOMWidget):
    def __init__(self, X_train, y_train, dataset_name):
        self.X_train = X_train
        self.y_train = y_train
        self.dataset_name = dataset_name
        self.current_algo = ''
        self.current_hp_box = None
        self.visualisation_types = None #do not delete
        self.algos_visualisation_type = None #do not delete
        self.action = ''
        self.test_acc = 0
        self.from_training = False 
        self.from_visualisation = False 
        
        self.run_cl = widgets.Button(description='Confirm!', disabled=False, button_style='info',
                                  tooltip='Click to confirm', icon='check')
        self.run_reg = widgets.Button(description='Confirm!', disabled=False, button_style='info',
                                  tooltip='Click to confirm', icon='check')
        self.run2 = widgets.Button(description='Confirm!', disabled=False, button_style='info',
                                  tooltip='Click to confirm', icon='check')
        self.button_plot = widgets.Button(description='Plot!', disabled=False, button_style='info', tooltip='Click to Plot', icon='pencil')
        self.go_to_visualisation_button = widgets.Button(description='Go to Visualisation', disabled=False,
                                                        button_style='info', tooltip='Click to see more', icon='bar-chart')
        self.go_to_training_button_1 = widgets.Button(description='Go to Training', disabled=False,
                                                        button_style='info', tooltip='Click to train more', icon='signal')
        self.go_to_training_button_2 = widgets.Button(description='Go to Training', disabled=False,
                                                        button_style='info', tooltip='Click to train more', icon='signal')

        
        self.algo_action = 2
        self.algo_level = 4 
        self.guidance_level = 6
        self.param_level = 8
        self.training_level = 10
        self.plotting_level = 12

        self.classification_algos_visualisation = [
            Button(description='Random Forest', layout=Layout(flex='4 1 auto', width='auto')),
            Button(description='knn', layout=Layout(flex='4 1 auto', width='auto')),
            Button(description='SVM', layout=Layout(flex='4 1 auto', width='auto')),
            Button(description='Overall', layout=Layout(flex='4 1 auto', width='auto'))
        ]

        self.regression_algos_visualisation = [
            Button(description='Random Forest', layout=Layout(flex='4 1 auto', width='auto')),
            Button(description='Linear Regression', layout=Layout(flex='4 1 auto', width='auto')),
            Button(description='Logistic Regression', layout=Layout(flex='4 1 auto', width='auto')),
            Button(description='Overall', layout=Layout(flex='4 1 auto', width='auto'))
        ]

        self.regression_algos_training = [
            Button(description='Random Forest', layout=Layout(flex='3 1 auto', width='auto')),
            Button(description='Linear Regression', layout=Layout(flex='3 1 auto', width='auto')),
            Button(description='Logistic Regression', layout=Layout(flex='3 1 auto', width='auto'))
        ]

        self.classification_algos_training = [
            Button(description='Random Forest', layout=Layout(flex='3 1 auto', width='auto')),
            Button(description='knn', layout=Layout(flex='3 1 auto', width='auto')),
            Button(description='SVM', layout=Layout(flex='3 1 auto', width='auto'))
        ]
        
    def init(self):
        action_question = widgets.HTML('<h1>What do you want to run?</h1>')
        
        self.actions = [
            Button(description='Training', layout=Layout(flex='3 1%', width='auto')),
            Button(description='Visualisation', layout=Layout(flex='3 1%', width='auto'))                                       ]
        
        for action in self.actions:
            action.on_click(self.show_actions)
            
        self.box_layout = Layout(display='flex', flex_flow='row', aligh_items='stretch', width='100%')
        actions_box = Box(children=self.actions, layout=self.box_layout)
        
        self.container=VBox([action_question, actions_box])
        display(self.container)

    def show_actions(self, button):
        for btn in self.actions:
            btn.style.button_color = 'lightgray'
        button.style.button_color = 'lightblue'

        type_question = widgets.HTML('<h2>What task do you want to do?</h2>')
        self.container.children = tuple(list(self.container.children)[:self.algo_level] + [type_question])
        
        action_box=Box()
        
        if button.description=='Training':
            self.action = 'training'
            action_box=self.get_types_training()

        elif button.description=='Visualisation':
            self.action = 'visualisation'
            action_box=self.get_types_visualisation()

        self.container.children = tuple(list(self.container.children)[:self.algo_action+1]+[action_box])

    def get_types_training(self):
        self.from_visualisation = False
        self.action = 'training'
        self.ml_types = [
            Button(description='Classification', layout=Layout(flex='2 1 0%', width='auto')),
            Button(description='Regression', layout=Layout(flex='2 1 0%', width='auto'))
        ]
        for training_type in self.ml_types:
            training_type.on_click(self.show_types)

        return Box(children=self.ml_types, layout=self.box_layout)

    
    def get_types_visualisation(self):
        self.from_training = False
        self.action = 'visualisation'
        self.ml_types = [
            Button(description='Classification', layout=Layout(flex='2 1 0%', width='auto')),
            Button(description='Regression', layout=Layout(flex='2 1 0%', width='auto'))
        ]
        
        for training_type in self.ml_types:
            training_type.on_click(self.show_types)

        return Box(children=self.ml_types, layout=self.box_layout)


    def show_types(self, button):
        if button.description == 'Go to Training':
            self.action = 'training'
            self.go_to_training_button_2 = widgets.Button(description='Go to Training', disabled=False,
                                                        button_style='info', tooltip='Click to train more', icon='signal')
            for btn in self.actions:
                if btn.description == 'Training':
                    btn.style.button_color = 'lightblue'
                else:
                    btn.style.button_color = 'lightgray'
        
            for btn in self.ml_types:
                if btn.description == self.alg_type:
                    btn.style.button_color = 'lightblue'
                else:
                    btn.style.button_color = 'lightgray'
            if self.action == 'training':
                algo_question = widgets.HTML('<h3>Which {} algorithm do you want to train?</h3>'.format(self.alg_type))
            elif self.action == 'visualisation':
                algo_question = widgets.HTML('<h3>Which {} algorithm do you want to visualise?</h3>'.format(self.alg_type))
            self.container.children = tuple(list(self.container.children)[:self.algo_level] + [algo_question])

            if self.alg_type == 'Classification':
                self.alg_type = 'Classification'
                algo_box = self.get_classification_algos()

            if self.alg_type == 'Regression':
                self.alg_type = 'Regression'
                algo_box = self.get_regression_algos()
            
            self.from_visualisation = False 
            self.container.children = tuple(list(self.container.children)[:self.algo_level+1] + [algo_box])

        else:
        
            for btn in self.ml_types:
                btn.style.button_color = 'lightgray'
            button.style.button_color = 'lightblue'
            if self.action == 'training':
                algo_question = widgets.HTML('<h3>Which {} algorithm do you want to train?</h3>'.format(button.description))
            elif self.action == 'visualisation':
                algo_question = widgets.HTML('<h3>Which {} algorithm do you want to visualise?</h3>'.format(button.description))
            self.container.children = tuple(list(self.container.children)[:self.algo_level] + [algo_question])
            
            algo_box = Box()
    
            if button.description == 'Classification':
                self.alg_type = 'Classification'
                algo_box = self.get_classification_algos()
            elif button.description == 'Regression':
                self.alg_type = 'Regression'
                algo_box = self.get_regression_algos()
                
            self.container.children = tuple(list(self.container.children)[:self.algo_level+1] + [algo_box])
        
    def get_classification_algos(self):
        if self.action == 'training':
            self.classification_algos_training = [
                Button(description='Random Forest', layout=Layout(flex='3 1 auto', width='auto')),
                Button(description='knn', layout=Layout(flex='3 1 auto', width='auto')),
                Button(description='SVM', layout=Layout(flex='3 1 auto', width='auto'))
            ]
            for algo in self.classification_algos_training:
                algo.on_click(self.show_algos)
            return Box(children=self.classification_algos_training, layout=self.box_layout)
        
        elif self.action == 'visualisation':
            self.classification_algos_visualisation = [
                Button(description='Random Forest', layout=Layout(flex='4 1 auto', width='auto')),
                Button(description='knn', layout=Layout(flex='4 1 auto', width='auto')),
                Button(description='SVM', layout=Layout(flex='4 1 auto', width='auto')),
                Button(description='Overall', layout=Layout(flex='4 1 auto', width='auto'))
            ]
            for algo in self.classification_algos_visualisation:
                algo.on_click(self.classification_visualisation_types)
            return Box(children=self.classification_algos_visualisation, layout=self.box_layout)

    def get_regression_algos(self):
        if self.action == 'training':
            self.regression_algos_training = [
                Button(description='Random Forest', layout=Layout(flex='3 1 auto', width='auto')),
                Button(description='Linear Regression', layout=Layout(flex='3 1 auto', width='auto')),
                Button(description='Logistic Regression', layout=Layout(flex='3 1 auto', width='auto'))
            ]

            for algo in self.regression_algos_training:
                algo.on_click(self.show_algos)
            return Box(children=self.regression_algos_training, layout=self.box_layout)
        
        elif self.action == 'visualisation':
            self.regression_algos_visualisation = [
                Button(description='Random Forest', layout=Layout(flex='4 1 auto', width='auto')),
                Button(description='Linear Regression', layout=Layout(flex='4 1 auto', width='auto')),
                Button(description='Logistic Regression', layout=Layout(flex='4 1 auto', width='auto')),
                Button(description='Overall', layout=Layout(flex='4 1 auto', width='auto'))
            ]

            for algo in self.regression_algos_visualisation:
                algo.on_click(self.regression_visualisation_types)
            return Box(children=self.regression_algos_visualisation, layout=self.box_layout)

    def show_algos(self, button):
        self.action = 'training'
        if button is None:
            for btn in self.get_current_algo_btns():
                if btn.description == self.current_algo:
                    btn.style.button_color = 'lightblue'
                else:
                    btn.style.button_color = 'lightgray'
            alg = self.current_algo

        elif button.description == 'Go to Training':
            self.go_to_training_button_1 = widgets.Button(description='Go to Training', disabled=False,
                                                        button_style='info', tooltip='Click to train more', icon='signal')

            alg = self.algos_visualisation_type

            for btn in self.actions:
                if btn.description == 'Training':
                    btn.style.button_color = 'lightblue'
                else:
                    btn.style.button_color = 'lightgray'

            for btn in self.ml_types:
                if btn.description == self.alg_type:
                    btn.style.button_color = 'lightblue'
                else:
                    btn.style.button_color = 'lightgray'

            if self.alg_type == 'Classification':
                algo_box = self.get_classification_algos()

            if self.alg_type == 'Regression':
                algo_box = self.get_regression_algos()

            algo_question = widgets.HTML('<h3>Which {} algorithm do you want to train?</h3>'.format(self.alg_type))
            self.container.children = tuple(list(self.container.children)[:self.algo_level] +[algo_question] + [algo_box])
            alg = self.current_algo
            self.from_visualisation = False
            self.show_algos(button=None)
            
        else:

            for btn in self.get_current_algo_btns():
                btn.style.button_color = 'lightgray'
            button.style.button_color = 'lightblue'

            alg = button.description

        
        guidance_question = widgets.HTML('<h4>In which mode do you prefer to train {}?</h4>'.format(alg))
        self.container.children = tuple(list(self.container.children)[:self.guidance_level] + [guidance_question])
                
        self.guidance_types = [Button(description='Default', layout=Layout(flex='3 1 auto', width='auto')),
                                Button(description='Supported', layout=Layout(flex='3 1 auto', width='auto')),
                                Button(description='Profi', layout=Layout(flex='3 1 auto', width='auto'))]
        
        guidance_box = Box(children=self.guidance_types, layout=self.box_layout)
        for btn in self.guidance_types:
            btn.on_click(self.show_hyperparamters)
        self.container.children = tuple(list(self.container.children)[:self.guidance_level+1] + [guidance_box])

    def classification_visualisation_types(self, button):
        for btn in self.classification_algos_visualisation:
            btn.style.button_color = 'lightgray'
        button.style.button_color = 'lightblue'
        self.algos_visualisation_type = button.description
        if self.algos_visualisation_type != "Overall":
            self.current_algo = button.description
            guidance_question = widgets.HTML('<h4>In which mode do you prefer to visualise {}?</h4>'.format(button.description))
            self.container.children = tuple(list(self.container.children)[:self.guidance_level] + [guidance_question])
                    
            self.visualisation_types = [Button(description='Hyperparameter Importance Plot', layout=Layout(flex='3 1 auto', width='auto')),
                                    Button(description='Parallel Coordinate Hyperparameter Plot', layout=Layout(flex='3 1 auto', width='auto'))]
            
            visualisation_box = Box(children=self.visualisation_types, layout=self.box_layout)
            for btn in self.visualisation_types:
                btn.on_click(self.show_visualisation)
            self.container.children = tuple(list(self.container.children)[:self.guidance_level+1] + [visualisation_box])
        else:
            self.show_visualisation(button)

    def regression_visualisation_types(self, button):
        for btn in self.regression_algos_visualisation:
            btn.style.button_color = 'lightgray'
        button.style.button_color = 'lightblue'
        self.algos_visualisation_type = button.description
        if self.algos_visualisation_type != "Overall":
            self.current_algo = button.description
            guidance_question = widgets.HTML('<h4>In which mode do you prefer to visualise {}?</h4>'.format(button.description))
            self.container.children = tuple(list(self.container.children)[:self.guidance_level] + [guidance_question])
                    
            self.visualisation_types = [Button(description='Hyperparameter Importance Plot', layout=Layout(flex='3 1 auto', width='auto')),
                                    Button(description='Parallel Coordinate Hyperparameter Plot', layout=Layout(flex='3 1 auto', width='auto'))]
            
            visualisation_box = Box(children=self.visualisation_types, layout=self.box_layout)
            for btn in self.visualisation_types:
                btn.on_click(self.show_visualisation)
            self.container.children = tuple(list(self.container.children)[:self.guidance_level+1] + [visualisation_box])
        else:
            self.show_visualisation(button)
    
    def scatter_plot_overview_helper(self, button):
        first = int(self.chosen_parameters.children[0].value)
        second = int(self.chosen_parameters.children[1].value)
        third = int(self.chosen_parameters.children[2].value)
        n_chosen= first+second+third
        if n_chosen != 2: 
            self.container.children = tuple(list(self.container.children)[:self.guidance_level+3] +
                                                [widgets.HTML('<h4>Not two selected. Please select 2!</h4>')])
        else:
            fig_widget = scatter_plot_overview(self.dataset_name, self.alg_type, first, second, third)
            self.container.children = tuple(list(self.container.children)[:self.guidance_level+3] +
                                                fig_widget+ [self.go_to_training_button_2])
            self.from_visualisation = True
            self.go_to_training_button_2.on_click(self.show_types)
            

    def show_visualisation(self, button):
        self.action = 'visualisation'
        if self.visualisation_types is not None and self.algos_visualisation_type != "Overall":
            for btn in self.visualisation_types:
                btn.style.button_color = 'lightgray'
            button.style.button_color = 'lightblue'

        if button.description == "Overall" or button.description == 'Go to Visualisation':
            if button.description == 'Go to Visualisation':
                for btn in self.actions:
                    if btn.description == 'Visualisation':
                        btn.style.button_color = 'lightblue'
                    else:
                        btn.style.button_color = 'lightgray'
                for btn in self.ml_types:
                    if btn.description == self.alg_type:
                        btn.style.button_color = 'lightblue'
                    else:
                        btn.style.button_color = 'lightgray'
                if self.alg_type == 'Classification':
                    algos_visualisation =  self.classification_algos_visualisation
                else:
                    algos_visualisation =  self.regression_algos_visualisation
                algo_question = widgets.HTML('<h3>Which {} algorithm do you want to visualise?</h3>'.format(self.alg_type))
                self.container.children = tuple(list(self.container.children)[:self.algo_level]+[algo_question] + [Box(children=algos_visualisation, layout=self.box_layout)])
                for btn in algos_visualisation:
                    if btn.description == 'Overall':
                        btn.style.button_color = 'lightblue'
                    else:
                        btn.style.button_color = 'lightgray'
                for algo in algos_visualisation:
                    if self.alg_type == 'Classification':
                        algo.on_click(self.classification_visualisation_types)
                    else:
                        algo.on_click(self.regression_visualisation_types)

                self.from_training = False
        
            guidance_question = widgets.HTML('<h4>Please select two metrics:')
            self.container.children = tuple(list(self.container.children)[:self.guidance_level] + [guidance_question])
            self.chosen_parameters = self.create_box_scatter_visualisation_choice()
            self.container.children = tuple(list(self.container.children)[:self.guidance_level+1] + [self.chosen_parameters])
            first = int(self.chosen_parameters.children[0].value)
            second = int(self.chosen_parameters.children[1].value)
            third = int(self.chosen_parameters.children[2].value)
            self.container.children = tuple(list(self.container.children)[:self.guidance_level+2] + [self.run2])
            fig_widget = scatter_plot_overview(self.dataset_name, self.alg_type, first, second, third)
            self.container.children = tuple(list(self.container.children)[:self.guidance_level+3] +
                                                fig_widget+[self.go_to_training_button_2])

            self.run2.on_click(self.scatter_plot_overview_helper)
            self.from_visualisation = True
            self.go_to_training_button_2.on_click(self.show_types)
            

        elif self.algos_visualisation_type != "Overall" and button.description == 'Hyperparameter Importance Plot' and self.from_training == False:
            fig_widget = feature_importance(self.dataset_name, self.algos_visualisation_type, self.alg_type)
            self.container.children = tuple(list(self.container.children)[:self.guidance_level+2] + fig_widget+
                                            [self.go_to_training_button_1])
            self.from_visualisation = True
            self.go_to_training_button_1.on_click(self.show_algos)

        elif self.algos_visualisation_type != "Overall" and button.description == 'Parallel Coordinate Hyperparameter Plot':
            fig_widget = parallel_coordinates(self.dataset_name, self.algos_visualisation_type, self.alg_type)
            self.container.children = tuple(list(self.container.children)[:self.guidance_level+2] + fig_widget +
                                            [self.go_to_training_button_1])
            self.from_visualisation = True
            self.go_to_training_button_1.on_click(self.show_algos)
        
    
    def show_hyperparamters(self, button):
        for btn in self.guidance_types:
            btn.style.button_color = 'lightgray'
        button.style.button_color = 'lightblue'
        try:
            if self.get_active_btn(self.ml_types).description == 'Classification':
                self.show_classification_hyperparams(button)
            else:
                self.show_regression_hyperparams(button)
        except:
            if self.alg_type == 'Classification':
                self.show_classification_hyperparams(button)
            else:
                self.show_regression_hyperparams(button)
            
    def show_classification_hyperparams(self, button):
        try:
            if self.get_active_btn(self.classification_algos_training).description == 'Random Forest':
                self.show_rf_classification_hyperparams(button)
            elif self.get_active_btn(self.classification_algos_training).description == 'knn':
                self.show_knn_hyperparams(button)
            else:
                self.show_svm_params(button)
        except:
            if self.current_algo == 'Random Forest':
                self.show_rf_classification_hyperparams(button)
            elif self.current_algo == 'knn':
                self.show_knn_hyperparams(button)
            else:
                self.show_svm_params(button)
            
    def show_regression_hyperparams(self, button):
        try:
            if self.get_active_btn(self.regression_algos_training).description == 'Random Forest':
                self.show_rf_regression_hyperparams(button)
            elif self.get_active_btn(self.regression_algos_training).description == 'Linear Regression':
                self.show_lin_regression_hyperparams(button)
            else:
                self.show_log_regression_params(button)

        except: 
            if self.current_algo == 'Random Forest':
                self.show_rf_regression_hyperparams(button)
            elif self.current_algo == 'Linear Regression':
                self.show_lin_regression_hyperparams(button)
            else:
                self.show_log_regression_params(button)
            
    def show_rf_regression_hyperparams(self, button):
        if button.description == 'Default':
            self.current_algo = 'reg_rf_def'
            self.container.children = tuple(list(self.container.children)[:self.param_level] +
                                            [widgets.HTML("The following hyperparameters will be used for training. Please confirm. <br/>"
                                                          "{n_estimators: 100, <br/>criterion: 'squared_error', <br/>max_depth: None, <br/>"
                                                          "min_samples_split: 2, <br/>min_samples_leaf: 1, <br/>min_weight_fraction_leaf: 0.0, <br/>"
                                                          "max_features: 'auto', <br/>max_leaf_nodes: None, <br/>min_impurity_decrease: 0.0, <br/>"
                                                          "bootstrap: True, <br/>oob_score: False, <br/>warm_start: False, <br/>"
                                                          "ccp_alpha: 0.0, <br/>max_samples: None}")])
        elif button.description == 'Supported':
            self.current_algo = 'reg_rf_sup'
            self.current_hp_box = self.create_box_reg_rf_sup()
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [self.current_hp_box])
        elif button.description == 'Profi':
            self.current_algo = 'reg_rf_pro'
            self.current_hp_box = self.create_box_reg_rf_pro()
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [self.current_hp_box])
            
        self.container.children = tuple(list(self.container.children)[:self.param_level+1] + [self.run_reg])

        self.run_reg.on_click(self.reg_rf)

    def show_rf_classification_hyperparams(self, button):
        if button.description == 'Default':
            self.current_algo = 'class_rf_def'
            self.container.children = tuple(list(self.container.children)[:self.param_level] +
                                            [widgets.HTML("The following hyperparameters will be used for training. Please confirm. <br/>"
                                                          "{n_estimators: 100, <br/>criterion: 'gini', <br/>"
                                                          "max_depth: None, <br/>min_samples_split: 2, <br/>"
                                                          "min_samples_leaf: 1, <br/>min_weight_fraction_leaf: 0.0,<br/>"
                                                          "max_features: 'auto', <br/>max_leaf_nodes: None, <br/>"
                                                          "min_impurity_decrease: 0.0, <br/>bootstrap: True,<br/>"
                                                          "oob_score: False, <br/>warm_start: 'False', <br/>"
                                                          "class_weight: None, <br/>ccp_alpha: 0.0, <br/>max_samples: None}")])
        elif button.description == 'Supported':
            self.current_algo = 'class_rf_sup'
            self.current_hp_box = self.create_box_class_rf_sup()
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [self.current_hp_box])
        elif button.description == 'Profi':
            self.current_algo = 'class_rf_pro'
            self.current_hp_box = self.create_box_class_rf_pro()
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [self.current_hp_box])

        self.container.children = tuple(list(self.container.children)[:self.param_level+1] + [self.run_cl])

        self.run_cl.on_click(self.class_rf)

    def show_lin_regression_hyperparams(self, button):
        if button.description == 'Default':
            self.current_algo = 'reg_lin_def'
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [
                widgets.HTML("The following hyperparameters will be used for training. Please confirm. <br/>"
                             "{fit_intercept: True, <br/>normalize: False, <br/>copy_X: True, <br/>n_jobs: None,"
                             "<br/>positive: False}")])
        elif button.description == 'Supported':
            self.current_algo = 'reg_lin_sup'
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [
                widgets.HTML("The following hyperparameters (the same as default) will be used for training. Please confirm. <br/>"
                             "{fit_intercept: True, <br/>normalize: False, <br/>copy_X: True, <br/>n_jobs: None},"
                             "<br/>positive: False")])
        elif button.description == 'Profi':
            self.current_algo = 'reg_lin_pro'
            self.current_hp_box = self.create_box_reg_lin_pro()
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [self.current_hp_box])

        self.container.children = tuple(list(self.container.children)[:self.param_level + 1] + [self.run_reg])
        self.run_reg.on_click(self.reg_lin)
        
    def show_log_regression_params(self, button):
        if button.description == 'Default':
            self.current_algo = 'reg_log_def'
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [
                widgets.HTML("The following hyperparameters will be used for training. Please confirm. <br/>"
                             "{penalty: 'l2', <br/>dual: False, <br/>tol: 0.0001, <br/>C: 1.0, <br/>fit_intercept: True, <br/>"
                             "intercept_scaling: 1.0, <br/>class_weight: None, <br/>solver: 'lbfgs', <br/>max_iter: 100, <br/>"
                             "multi_class: 'auto', <br/>warm_start: False, <br/>l1_ratio: None}")])
        elif button.description == 'Supported':
            self.current_algo = 'reg_log_sup'
            self.current_hp_box = self.create_box_reg_log_sup()
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [self.current_hp_box])
        elif button.description == 'Profi':
            self.current_algo = 'reg_log_pro'
            self.current_hp_box = self.create_box_reg_log_pro()
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [self.current_hp_box])

        self.container.children = tuple(list(self.container.children)[:self.param_level + 1] + [self.run_reg])
        self.run_reg.on_click(self.reg_log)
        
    def show_knn_hyperparams(self, button):
        if button.description == 'Default':
            self.current_algo = 'class_knn_def'
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [
                widgets.HTML("The following hyperparameters will be used for training. Please confirm. <br/>"
                             "{n_neighbors: 5, <br/>weights: 'uniform', <br/>algorithm: 'auto', <br/>leaf_size: 30, <br/>"
                             "metric_params: None, <br/>p: 2, <br/>metric: 'minkowski', <br/>n_jobs: None}")])
        elif button.description == 'Supported':
            self.current_algo = 'class_knn_sup'
            self.current_hp_box = self.create_box_class_knn_sup()
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [self.current_hp_box])
        elif button.description == 'Profi':
            self.current_algo = 'class_knn_pro'
            self.current_hp_box = self.create_box_class_knn_pro()
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [self.current_hp_box])

        self.container.children = tuple(list(self.container.children)[:self.param_level + 1] + [self.run_cl])

        self.run_cl.on_click(self.class_knn)
        
    def show_svm_params(self, button):
        if button.description == 'Default':
            self.current_algo = 'class_svm_def'
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [
                widgets.HTML("The following hyperparameters will be used for training. Please confirm. <br/>"
                             "{C: 1.0, <br/>kernel: 'rbf', <br/>degree: 3, <br/>gamma: 'scale', <br/>coef0: 0.0, <br/>"
                             "shrinking: True, <br/>probability: False, <br/>tol: 0.001, <br/>cache_size: 200.0, <br/>"
                             "class_weight: None, <br/>max_iter: -1, <br/>verbose: False, <br/>"
                             "decision_function_shape: 'ovr', <br/>break_ties: False, <br/>random_state: None}")])
        elif button.description == 'Supported':
            self.current_algo = 'class_svm_sup'
            self.current_hp_box = self.create_box_class_svm_sup()
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [self.current_hp_box])
        elif button.description == 'Profi':
            self.current_algo = 'class_svm_pro'
            self.current_hp_box = self.create_box_class_svm_pro()
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [self.current_hp_box])

        self.container.children = tuple(list(self.container.children)[:self.param_level + 1] + [self.run_cl])

        self.run_cl.on_click(self.class_svm)

    def get_active_btn(self, btn_array):
        return [btn for btn in btn_array if btn.style.button_color == 'lightblue'][0]
    
    def get_current_algo_btns(self):
        if self.from_visualisation == True:
            if self.get_active_btn(self.ml_types).description == 'Classification':
                self.classification_algos_training =  [
                    Button(description='Random Forest', layout=Layout(flex='3 1 auto', width='auto')),
                    Button(description='knn', layout=Layout(flex='3 1 auto', width='auto')),
                    Button(description='SVM', layout=Layout(flex='3 1 auto', width='auto'))
                    ]
                return self.classification_algos_training
            elif self.get_active_btn(self.ml_types).description == 'Regression':
                self.regression_algos_training = [
                    Button(description='Random Forest', layout=Layout(flex='3 1 auto', width='auto')),
                    Button(description='Linear Regression', layout=Layout(flex='3 1 auto', width='auto')),
                    Button(description='Logistic Regression', layout=Layout(flex='3 1 auto', width='auto'))
                    ]
                return self.regression_algos_training
        
        elif self.get_active_btn(self.ml_types).description == 'Classification' and self.action == 'training':
            return self.classification_algos_training

        elif self.get_active_btn(self.ml_types).description == 'Classification' and self.action == 'visualisation':
            return self.classification_algos_visualisation

        elif self.get_active_btn(self.ml_types).description == 'Regression' and self.action == 'training':
            return self.regression_algos_training

        elif self.get_active_btn(self.ml_types).description == 'Regression' and self.action == 'visualisation':
            return self.regression_algos_visualisation

    def create_box_class_knn_sup(self):
        n_neighbors = widgets.IntSlider(min=1, max=len(self.X_train) / 2,
                                                      value=len(self.X_train) ** (1 / 2),
                                                      step=1, description="n-neighbors",
                                                      style={'description_width': 'initial'})
        def react(slider):
            n_neighbors.style.handle_color = 'green' if slider <= len(self.X_train) ** (
                        1 / 2) + 5 and slider >= (len(self.X_train) ** (1 / 2)) / 2 - 5 else 'red'

        box = interactive(react, slider=n_neighbors)
        return box

    def create_box_class_knn_pro(self):
        fields = {
            'n-neighbors': ('', 5),
            'weights': (['uniform', 'distance'], 'uniform'),
            'algorithm': (['auto', 'ball_tree', 'kd_tree', 'brute'], 'auto'),
            'leaf size': ('', 30),
            'p': ('int', 2),
            'metric': ('str', 'minkowski'),
            'n_jobs': ('', -1)}

        widget = {}
        widget['grid'] = widgets.GridspecLayout(1, 2)
        vbox_widgets_left = []
        vbox_widgets_right = []
        for hp_name, hp_tuple in fields.items():
            widget_descp = hp_name
            layout = widgets.Layout(width='215px')
            label = widgets.Label(widget_descp, layout=layout)
            if hp_name == 'n-neighbors':
                layout = widgets.Layout(width='70%')
                text_box = widgets.BoundedIntText(placeholder=hp_tuple[0], value=hp_tuple[1], min=1, max=len(self.X_train), layout=layout, disabled=False)
            if hp_name == 'leaf size' or hp_name == 'p' or hp_name == 'n_jobs':
                layout = widgets.Layout(width='70%')
                text_box = widgets.IntText(placeholder=hp_tuple[0], value=hp_tuple[1], layout=layout, disabled=False)
            elif hp_name == 'algorithm' or hp_name == 'weights':
                layout = widgets.Layout(width='70%')
                text_box = widgets.RadioButtons(options=hp_tuple[0], value=hp_tuple[1], layout=layout, disabled=False)
            elif  hp_name == 'metric':
                layout = widgets.Layout(width='70%')
                text_box = widgets.Text(placeholder=hp_tuple[0], value=hp_tuple[1], layout=layout, disabled=False)

            widget[hp_name] = widgets.HBox(children=[label, text_box])
            if hp_name == 'n-neighbors' or hp_name == 'algorithm' or hp_name == 'p':
                vbox_widgets_left.append(widget[hp_name])
            else:
                vbox_widgets_right.append(widget[hp_name])

        widget['grid'][0, 0] = widgets.VBox(children=vbox_widgets_left)
        widget['grid'][0, 1] = widgets.VBox(children=vbox_widgets_right)
        widget['grid'].grid_gap = '20px'
        children = [widget['grid']]
        box = widgets.VBox(children=children)
        return box

    def create_box_reg_lin_pro(self):
        fields = {
            'fit intercept': (True, ''),
            'normalize': (False, ''),
            'copy X': (True, ''),
            'n_jobs': ('', -1),
            'positive': (False, '')}

        widget = {}
        widget['grid'] = widgets.GridspecLayout(1, 2)
        vbox_widgets_left = []
        vbox_widgets_right = []
        for hp_name, hp_tuple in fields.items():
            widget_descp = hp_name
            layout = widgets.Layout(width='215px')
            label = widgets.Label(widget_descp, layout=layout)
            if hp_name == 'n_jobs':
                layout = widgets.Layout(width='70%')
                text_box = widgets.IntText(placeholder=hp_tuple[0], value=hp_tuple[1], layout=layout, disabled=False)
            else:
                layout = widgets.Layout(width='70%')
                text_box = widgets.Checkbox(value=hp_tuple[0], layout=layout, disabled=False)

            widget[hp_name] = widgets.HBox(children=[label, text_box])
            if hp_name == 'fit intercept' or hp_name == 'copy X' or hp_name == 'positive':
                vbox_widgets_left.append(widget[hp_name])
            else:
                vbox_widgets_right.append(widget[hp_name])

        widget['grid'][0, 0] = widgets.VBox(children=vbox_widgets_left)
        widget['grid'][0, 1] = widgets.VBox(children=vbox_widgets_right)
        widget['grid'].grid_gap = '20px'
        children = [widget['grid']]
        box = widgets.VBox(children=children)
        return box

    def create_box_reg_log_pro(self):
        fields = {
            'penalty': (['l1', 'l2', 'elasticnet', 'none'], 'l2'),
            'dual': (False, ''),
            'tol': ('', 0.0001),
            'C': ('', 1.0),
            'fit intercept': (True, ''),
            'intercept scaling': ('', 1.0),
            'class weight': (['balanced', None], 'balanced'),
            'random state': ('int or None', 'None'),
            'solver': (['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 'lbfgs'),
            'max iter': ('', 100),
            'multi class': (['auto', 'ovr', 'multinomial'], 'auto'),
            'verbose': (False, ''),
            'warm start': (False, ''),
            'n_jobs': ('int or None', 'None'),
            'l1 ration': ('float or None', 'None')}

        widget = {}
        widget['grid'] = widgets.GridspecLayout(1, 2)
        vbox_widgets_left = []
        vbox_widgets_right = []
        for hp_name, hp_tuple in fields.items():
            widget_descp = hp_name
            layout = widgets.Layout(width='215px')
            label = widgets.Label(widget_descp, layout=layout)
            if hp_name == 'max iter' or hp_name == 'n_jobs':
                layout = widgets.Layout(width='70%')
                text_box = widgets.IntText(placeholder=hp_tuple[0], value=hp_tuple[1], layout=layout, disabled=False)
            if hp_name == 'class weight' or hp_name == 'penalty' or hp_name == 'solver' or hp_name == 'multi class':
                layout = widgets.Layout(width='70%')
                text_box = widgets.RadioButtons(options=hp_tuple[0], layout=layout, disabled=False)
            if hp_name == 'random state' or hp_name == 'l1 ration':
                layout = widgets.Layout(width='70%')
                text_box = widgets.Text(placeholder=hp_tuple[0], value=hp_tuple[1], layout=layout, disabled=False)
            if hp_name == 'tol' or hp_name == 'C' or hp_name == 'intercept scaling':
                layout = widgets.Layout(width='70%')
                text_box = widgets.FloatText(placeholder=hp_tuple[0], value=hp_tuple[1], layout=layout, disabled=False,
                                             style={'description_width': 'initial'})
            if hp_name == 'dual' or hp_name == 'fit intercept' or hp_name == 'verbose' or hp_name == 'warm start':
                layout = widgets.Layout(width='70%')
                text_box = widgets.Checkbox(value=hp_tuple[0], layout=layout, disabled=False)

            widget[hp_name] = widgets.HBox(children=[label, text_box])
            if hp_name == 'penalty' or hp_name == 'fit intercept' or hp_name == 'class weight' or hp_name == 'solver' or hp_name == 'multi class':
                vbox_widgets_left.append(widget[hp_name])
            else:
                vbox_widgets_right.append(widget[hp_name])

        widget['grid'][0, 0] = widgets.VBox(children=vbox_widgets_left)
        widget['grid'][0, 1] = widgets.VBox(children=vbox_widgets_right)
        widget['grid'].grid_gap = '20px'

        children = [widget['grid']]

        box = widgets.VBox(children=children)
        return box

    def create_box_reg_log_sup(self):
        penalty = widgets.RadioButtons(options=['none', 'l2'], description="penalty")
        c = widgets.RadioButtons(options=[0.001, 0.01, 0.1, 1, 10.0, 100.0, 1000.0], description="C")

        box_reg_log_sup = widgets.HBox(children=[penalty, c])
        return box_reg_log_sup

    def create_box_class_svm_pro(self):
        fields = {
            'C': ('', 1.0),
            'kernel': (['linear','poly','rbf','sigmoid'], 'rbf'),
            'degree': ('', 3),
            'gamma': ('scale, auto or float', 'scale'),
            'coef0': ('', 0.0),
            'shrinking': (True, ''),
            'probability': (False,''),
            'tol': ('', 0.001),
            'cache size': ('', 200.0),
            'class weight': (['balanced', None], 'balanced'),
            'verbose': (False, ''),
            'max iter': ('', -1),
            'decision function shape': (['ovo','ovr'],'ovr'),
            'break ties': (False, ''),
            'random state': ('int or None', 'None')}

        widget = {}
        widget['grid'] = widgets.GridspecLayout(1, 2)
        vbox_widgets_left = []
        vbox_widgets_right = []
        for hp_name, hp_tuple in fields.items():
            widget_descp = hp_name
            layout = widgets.Layout(width='215px')
            label = widgets.Label(widget_descp, layout=layout)
            if hp_name == 'degree' or hp_name == 'max iter':
                layout = widgets.Layout(width='70%')
                text_box = widgets.IntText(placeholder=hp_tuple[0], value = hp_tuple[1], layout=layout, disabled=False)
            if hp_name == 'kernel' or hp_name == 'decision function shape' or hp_name == 'class weight':
                layout = widgets.Layout(width='70%')
                text_box = widgets.RadioButtons(options=hp_tuple[0], value=hp_tuple[1], layout=layout, disabled=False)
            if hp_name == 'gamma' or hp_name == 'random state':
                layout = widgets.Layout(width='70%')
                text_box = widgets.Text(placeholder=hp_tuple[0], value=hp_tuple[1], layout=layout, disabled=False)
            if hp_name == 'C' or hp_name == 'coef0' or hp_name == 'tol' or hp_name == 'cache size':
                layout = widgets.Layout(width='70%')
                text_box = widgets.FloatText(placeholder=hp_tuple[0], value=hp_tuple[1], layout=layout, disabled=False, style = {'description_width': 'initial'})
            if hp_name == 'shrinking' or hp_name == 'probability' or hp_name == 'verbose' or hp_name=='break ties':
                layout = widgets.Layout(width='70%')
                text_box = widgets.Checkbox(value=hp_tuple[0], layout=layout, disabled=False)

            widget[hp_name] = widgets.HBox(children=[label, text_box])
            if hp_name == 'C' or hp_name == 'degree' or hp_name == 'gamma' or hp_name == 'coef0' or hp_name == 'probability' or hp_name == 'cache size' or hp_name == 'verbose' or hp_name == 'decision function shape':
                vbox_widgets_left.append(widget[hp_name])
            else:
                vbox_widgets_right.append(widget[hp_name])

        widget['grid'][0, 0] = widgets.VBox(children=vbox_widgets_left)
        widget['grid'][0, 1] = widgets.VBox(children=vbox_widgets_right)
        widget['grid'].grid_gap = '20px'
        children = [widget['grid']]
        box = widgets.VBox(children = children)
        return box

    def create_box_class_svm_sup(self):
        c = widgets.RadioButtons(options=[0.001,0.01,0.1,1.0,10.0,100.0,1000.0], value=1, description="C")
        kernel = widgets.RadioButtons(options=['rbf','poly'], value='rbf', description="kernel")
        gamma = widgets.RadioButtons(options=['scale',0.001,0.01,0.1,1.0,10.0,100.0,1000.0], value='scale', description="gamma")

        box = widgets.HBox(children=[c, kernel, gamma])
        return box

    def create_box_class_rf_pro(self):
        fields = {
            'n-estimators': ('', 1),
            'criterion': (['gini','entropy'], 'gini'),
            'max depth': ('int or None', 'None'),
            'min samples split': ('int or float in range (0.0, 1.0]', 0.1),
            'min samples leaf': ('int or float in range (0, 0.5]', 0.1),
            'min weight fraction leaf': ('float in range [0, 0.5]', 0),
            'max features': ('auto, sqrt, log2, int or float','auto'),
            'max leaf nodes': ('int or None','None'),
            'min impurity decrease': ('', 0) ,
            'bootstrap': (True,''),
            'oob score': (False,''),
            'n_jobs': ('', -1),
            'verbose': ('int', 0),
            'warm start': (False,''),
            'class weight': (['balanced','balanced_subsample',None], 'balanced'),
            'ccp alpha': ('', 0),
            'max samples': ('float in range (0,1] or None', 'None'),
            'random state': ('int or None', 'None')}

        widget = {}
        widget['grid'] = widgets.GridspecLayout(1, 2)
        vbox_widgets_left = []
        vbox_widgets_right = []
        for hp_name, hp_tuple in fields.items():
            widget_descp = hp_name
            layout = widgets.Layout(width='215px')
            label = widgets.Label(widget_descp, layout=layout)
            if hp_name == 'n-estimators' or hp_name == 'n_jobs' or hp_name == 'verbose':
                layout = widgets.Layout(width='70%')
                text_box = widgets.IntText(placeholder=hp_tuple[0], value = hp_tuple[1], layout=layout, disabled=False)
            if hp_name == 'criterion' or hp_name == 'class weight':
                layout = widgets.Layout(width='70%')
                text_box = widgets.RadioButtons(options=hp_tuple[0], value = hp_tuple[1], layout=layout, disabled=False)
            if hp_name == 'max depth' or hp_name == 'max features' or hp_name == 'max leaf nodes' or hp_name == 'max samples' or hp_name == 'random state':
                layout = widgets.Layout(width='70%')
                text_box = widgets.Text(placeholder=hp_tuple[0], value = hp_tuple[1], layout=layout, disabled=False)
            if hp_name == 'min samples split' or hp_name == 'min samples leaf' or hp_name == 'min impurity decrease' or hp_name == 'ccp alpha':
                layout = widgets.Layout(width='70%')
                text_box = widgets.FloatText(placeholder=hp_tuple[0], value=hp_tuple[1], layout=layout, disabled=False, style = {'description_width': 'initial'})
            if hp_name == 'min weight fraction leaf':
                layout = widgets.Layout(width='70%')
                text_box = widgets.BoundedFloatText(placeholder=hp_tuple[0], value=hp_tuple[1], min=0, max=0.5, step=0.001, layout=layout, disabled=False, style = {'description_width': 'initial'})
            if hp_name == 'bootstrap' or hp_name == 'oob score' or hp_name == 'warm start':
                layout = widgets.Layout(width='70%')
                text_box = widgets.Checkbox(value=hp_tuple[0], layout=layout, disabled=False)

            widget[hp_name] = widgets.HBox(children=[label, text_box])
            if hp_name == 'n-estimators' or hp_name == 'max depth' or hp_name == 'min samples leaf' or hp_name == 'max features' or hp_name == 'min impurity decrease' or hp_name == 'oob score' or hp_name == 'verbose' or hp_name == 'class weight' or hp_name == 'max samples':
                vbox_widgets_left.append(widget[hp_name])
            else:
                vbox_widgets_right.append(widget[hp_name])

        widget['grid'][0, 0] = widgets.VBox(children=vbox_widgets_left)
        widget['grid'][0, 1] = widgets.VBox(children=vbox_widgets_right)
        widget['grid'].grid_gap = '20px'
        children = [widget['grid']]
        box = widgets.VBox(children = children)
        return box

    def create_box_class_rf_sup(self):
        n_estimators = widgets.RadioButtons(options=[50, 100, 500, 1000], value=50, description="n-estimators", style={'description_width': 'initial'})
        max_depth  = widgets.IntSlider(min=1, max=len(self.X_train) * 2, value=len(self.X_train), step=1, description="max depth", style={'description_width': 'initial'})
        max_features = widgets.IntSlider(min=1, max=len(self.X_train[0]), value=len(self.X_train[0]) ** 0.5, step=1, description="max features", style={'description_width': 'initial'})

        def react_1(slider_1):
            max_depth .style.handle_color = 'green' if slider_1 <= len(self.X_train) + len(self.X_train) / 10 and slider_1 >= len(self.X_train) - len(self.X_train) / 10 else 'red'

        def react_2(slider_2):
            max_features.style.handle_color = 'green' if slider_2 <= len(self.X_train[0]) ** 0.5 + 1 and slider_2 >= len(self.X_train[0]) ** 0.5 - 1 else 'red'

        box_1 = interactive(react_1, slider_1=max_depth )
        box_2 = interactive(react_2, slider_2=max_features)

        box = widgets.HBox(children=[n_estimators, box_1, box_2])
        return box

    def create_box_reg_rf_pro(self):
        fields = {
            'n-estimators': ('', 1),
            'criterion': (['mse','mae'], 'mse'),
            'max depth': ('int or None', 'None'),
            'min samples split': ('int or float in range (0.0, 1.0]', 0.1),
            'min samples leaf': ('int or float in range (0, 0.5]', 0.1),
            'min weight fraction leaf': ('float in range [0, 0.5]', 0),
            'max features': ('auto, sqrt, log2, int or float','auto'),
            'max leaf nodes': ('int or None', 'None'),
            'min impurity decrease': ('', 0.0),
            'bootstrap': (True,''),
            'oob score': (False,''),
            'n_jobs': ('', -1),
            'verbose': ('', 0),
            'warm start': (False,''),
            'ccp alpha': ('', 0.0),
            'max samples': ('float in range (0,1] or None', 'None'),
            'random state': ('int or None', 'None')
            }

        widget = {}
        widget['grid'] = widgets.GridspecLayout(1, 2)
        vbox_widgets_left = []
        vbox_widgets_right = []
        for hp_name, hp_tuple in fields.items():
            widget_descp = hp_name
            layout = widgets.Layout(width='215px')
            label = widgets.Label(widget_descp, layout=layout)
            if hp_name == 'n-estimators' or hp_name == 'n_jobs' or hp_name == 'verbose':
                layout = widgets.Layout(width='70%')
                text_box = widgets.IntText(placeholder=hp_tuple[0], value = hp_tuple[1], layout=layout, disabled=False)
            if hp_name == 'criterion':
                layout = widgets.Layout(width='70%')
                text_box = widgets.RadioButtons(options=hp_tuple[0], layout=layout, disabled=False)
            if hp_name == 'max depth' or hp_name == 'max features' or hp_name == 'max leaf nodes' or hp_name == 'max samples' or hp_name == 'random state':
                layout = widgets.Layout(width='70%')
                text_box = widgets.Text(placeholder=hp_tuple[0], value = hp_tuple[1], layout=layout, disabled=False)
            if hp_name == 'min samples split' or hp_name == 'min samples leaf' or hp_name == 'min impurity decrease' or hp_name == 'ccp alpha':
                layout = widgets.Layout(width='70%')
                text_box = widgets.FloatText(placeholder=hp_tuple[0], value=hp_tuple[1], layout=layout, disabled=False, style = {'description_width': 'initial'})
            if hp_name == 'min weight fraction leaf':
                layout = widgets.Layout(width='70%')
                text_box = widgets.BoundedFloatText(placeholder=hp_tuple[0], value=hp_tuple[1], min=0, max=0.5, step=0.001, layout=layout, disabled=False, style = {'description_width': 'initial'})
            if hp_name == 'bootstrap' or hp_name == 'oob score' or hp_name == 'warm start':
                layout = widgets.Layout(width='70%')
                text_box = widgets.Checkbox(value=hp_tuple[0], layout=layout, disabled=False)

            widget[hp_name] = widgets.HBox(children=[label, text_box])
            if hp_name == 'n-estimators' or hp_name == 'max depth' or hp_name == 'min samples split' or hp_name == 'min weight fraction leaf' or hp_name == 'max leaf nodes' or hp_name == 'bootstrap' or hp_name == 'n_jobs' or hp_name == 'warm start' or hp_name == 'max samples':
                vbox_widgets_left.append(widget[hp_name])
            else:
                vbox_widgets_right.append(widget[hp_name])

        widget['grid'][0, 0] = widgets.VBox(children=vbox_widgets_left)
        widget['grid'][0, 1] = widgets.VBox(children=vbox_widgets_right)
        widget['grid'].grid_gap = '20px'
        children = [widget['grid']]
        box = widgets.VBox(children = children)
        return box

    def create_box_reg_rf_sup(self):
        n_estimators = widgets.RadioButtons(options=[50,100,500,1000], value =50, description="n-estimators", style={'description_width': 'initial'})
        max_depth = widgets.IntSlider(min=1, max=len(self.X_train)*2, value=len(self.X_train), step=1, description="max depth", style={'description_width': 'initial'})
        max_features = widgets.IntSlider(min=1, max=len(self.X_train[0]), value=len(self.X_train[0])/3, step=1,  description="max features", style={'description_width': 'initial'})

        def react_1(slider_1):
            max_depth.style.handle_color = 'green' if slider_1<=len(self.X_train)+len(self.X_train)/10 and slider_1>=len(self.X_train)-len(self.X_train)/10  else 'red'

        def react_2(slider_2):
            max_features.style.handle_color = 'green' if slider_2<=len(self.X_train[0]) and slider_2>=len(self.X_train[0])/3-1  else 'red'

        box_1 = interactive(react_1, slider_1=max_depth)
        box_2 = interactive(react_2, slider_2=max_features)

        box = widgets.HBox(children=[n_estimators, box_1, box_2])
        return box

    def create_box_scatter_visualisation_choice(self):
        choices = []
        if self.alg_type == 'Classification':
            selections = ['Test accuracy', 'Train accuracy', 'Fitting time']
        else:
            selections = ['Test neg-MSE', 'Train neg-MSE', 'Fitting time']
        for s in selections:
            if s == 'Fitting time':
                choices.append(widgets.Checkbox(value=False, description=s, disabled=False, indent=False))
            else: 
                choices.append(widgets.Checkbox(value=True, description=s, disabled=False, indent=False))

        box = widgets.HBox(children=choices)
        return box


    def class_knn(self, run):
        if self.current_algo == 'class_knn_def':
            filename = 'class_knn.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                current_df = pd.read_csv(fullname ,sep=';')

            else:
                current_df = pd.DataFrame(columns = ['n_neighbors', 'weights', 'algorithm', 'leaf_size',
                                                           'p', 'metric', 'n_jobs', 'Mean CV train acc', 'Mean CV test acc',
                                                          'Mean CV fit time'])

            trained_before_df = current_df.loc[(current_df['n_neighbors']==5)&
                                                  (current_df['weights']=='uniform')&
                                                  (current_df['algorithm']=='auto')&
                                                  (current_df['leaf_size']==30)&
                                                  (current_df['p']==2)&
                                                  (current_df['metric']=='minkowski')]
            if len(trained_before_df) > 0:
                result = widgets.Output()
                with result:
                    display(trained_before_df)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_current_df = widgets.HBox([result], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                                [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_current_df])
                self.test_acc = trained_before_df['Mean CV test acc'].values[0]

            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                clf = KNeighborsClassifier()
                scores = cross_validate(clf, self.X_train, self.y_train, cv=5, scoring='balanced_accuracy', return_train_score=True)
                train_acc = scores['train_score'].mean()
                self.test_acc = scores['test_score'].mean()
                fit_time = scores['fit_time'].mean()
                if len(current_df) > 0:
                    if self.test_acc > current_df['Mean CV test acc'].max():
                        result = f'Mean CV train accuracy: {round(train_acc,2)*100}%. <br/>Mean CV test accuracy:  {round(self.test_acc,2)*100}%. <br/>Best mean CV test accuracy for KNN so far!'   
                    else:
                        result = f'Mean CV train accuracy: {round(train_acc,2)*100}%. <br/>Mean CV test accuracy:  {round(self.test_acc,2)*100}%.'
                else:
                    result = f'Mean CV train accuracy: {round(train_acc,2)*100}%. <br/>Mean CV test accuracy:  {round(self.test_acc,2)*100}%. <br/>First KNN model trained!'
                        
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is: <br/>'+ result)])

                current_df = current_df.append({'n_neighbors': 5, 'weights': 'uniform',
                                                            'algorithm': 'auto', 'leaf_size': 30,
                                                            'p': 2, 'metric': 'minkowski', 'n_jobs': str(None), 
                                                            'Mean CV train acc': round(train_acc, 5),
                                                            'Mean CV test acc': round(self.test_acc, 5),
                                                            'Mean CV fit time': round(fit_time, 5)}, ignore_index = True)
                current_df.to_csv(fullname, sep=';', index=False)
     

        elif self.current_algo == 'class_knn_sup':
            filename = 'class_knn.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                current_df = pd.read_csv(fullname ,sep=';')

            else:
                current_df = pd.DataFrame(columns = ['n_neighbors', 'weights', 'algorithm', 'leaf_size',
                                                           'p', 'metric', 'n_jobs', 'Mean CV train acc', 'Mean CV test acc',
                                                          'Mean CV fit time'])

            n_neighbors = self.current_hp_box.children[0].value

            trained_before_df = current_df.loc[(current_df['n_neighbors']==n_neighbors)]
            if len(trained_before_df) > 0:
                result = widgets.Output()
                with result:
                    display(trained_before_df)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_current_df = widgets.HBox([result], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_current_df])
                
                self.test_acc = trained_before_df['Mean CV test acc'].values[0]

            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                try:
                    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
                    scores = cross_validate(clf, self.X_train, self.y_train, cv=5, scoring='balanced_accuracy', return_train_score=True)
                    train_acc = scores['train_score'].mean()
                    self.test_acc = scores['test_score'].mean()
                    fit_time = scores['fit_time'].mean()
                    
                except ValueError as err:
                    self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                    [widgets.HTML(str(err))])
                    return

                if len(current_df) > 0:
                    if self.test_acc > current_df['Mean CV test acc'].max():
                        result = f'Mean CV train accuracy: {round(train_acc,2)*100}%. <br/>Mean CV test accuracy:  {round(self.test_acc,2)*100}%. <br/>Best mean CV test accuracy for KNN so far!'
                    else:
                        result = f'Mean CV train accuracy: {round(train_acc,2)*100}%. <br/>Mean CV test accuracy:  {round(self.test_acc,2)*100}%.'
                else:
                    result = f'Mean CV train accuracy: {round(train_acc,2)*100}%. <br/>Mean CV test accuracy:  {round(self.test_acc,2)*100}%. <br/>First KNN model trained!'
                    
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is: <br/>'+ result)])

                current_df = current_df.append({'n_neighbors': n_neighbors, 'weights': 'uniform',
                                                            'algorithm': 'auto', 'leaf_size': 30,
                                                            'p': 2, 'metric': 'minkowski', 'n_jobs': str(None),
                                                            'Mean CV train acc': round(train_acc, 5),
                                                            'Mean CV test acc': round(self.test_acc, 5),
                                                            'Mean CV fit time': round(fit_time, 5)}, ignore_index = True)
                current_df.to_csv(fullname, sep=';', index=False)

        elif self.current_algo == 'class_knn_pro':
            filename = 'class_knn.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                current_df = pd.read_csv(fullname, sep=';')

            else:
                current_df = pd.DataFrame(columns=['n_neighbors', 'weights', 'algorithm', 'leaf_size',
                                                        'p', 'metric', 'Mean CV train acc', 'Mean CV test acc',
                                                          'Mean CV fit time'])

            n_neighbors= self.current_hp_box.children[0].children[0].children[0].children[1].value
            weights=self.current_hp_box.children[0].children[1].children[0].children[1].value
            algorithm=self.current_hp_box.children[0].children[0].children[1].children[1].value
            leaf_size= self.current_hp_box.children[0].children[1].children[1].children[1].value
            p= self.current_hp_box.children[0].children[0].children[2].children[1].value
            metric=self.current_hp_box.children[0].children[1].children[2].children[1].value
            n_jobs=self.current_hp_box.children[0].children[1].children[3].children[1].value

            trained_before_df = current_df.loc[(current_df['n_neighbors']==n_neighbors)&
                                                  (current_df['weights']==weights)&
                                                  (current_df['algorithm']==algorithm)&
                                                  (current_df['leaf_size']==leaf_size)&
                                                  (current_df['p']==p)&
                                                  (current_df['metric']==metric)]

            if len(trained_before_df) > 0:
                result = widgets.Output()
                with result:
                    display(trained_before_df)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_current_df = widgets.HBox([result], layout=box_layout)
                #self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                #[widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_current_df]+[self.button_plot])
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_current_df])

                self.test_acc = trained_before_df['Mean CV test acc'].values[0]

            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                try:
                    clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
                                                             leaf_size=leaf_size, p=p, metric=metric,n_jobs=n_jobs)
                    scores = cross_validate(clf, self.X_train, self.y_train, cv=5, scoring='balanced_accuracy', return_train_score=True)
                    train_acc = scores['train_score'].mean()
                    self.test_acc = scores['test_score'].mean()
                    fit_time = scores['fit_time'].mean()
                    
                except ValueError as err:
                    self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                    [widgets.HTML(str(err))])
                    return

                if len(current_df) > 0:
                    if self.test_acc > current_df['Mean CV test acc'].max():
                        result = f'Mean CV train accuracy: {round(train_acc,2)*100}%. <br/>Mean CV test accuracy:  {round(self.test_acc,2)*100}%. <br/>Best mean CV test accuracy for KNN so far!'
                    else:
                        result = f'Mean CV train accuracy: {round(train_acc,2)*100}%. <br/>Mean CV test accuracy:  {round(self.test_acc,2)*100}%.'
                else:
                    result = f'Mean CV train accuracy: {round(train_acc,2)*100}%. <br/>Mean CV test accuracy:  {round(self.test_acc,2)*100}%. <br/>First KNN model trained!'
                    
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is: <br/>'+ result)])
           
                current_df = current_df.append({'n_neighbors': n_neighbors, 'weights': weights,
                                                            'algorithm': algorithm, 'leaf_size': leaf_size,
                                                            'p': p, 'metric': metric, 'n_jobs': str(n_jobs),
                                                            'Mean CV train acc': round(train_acc, 5),
                                                            'Mean CV test acc': round(self.test_acc, 5),
                                                            'Mean CV fit time': round(fit_time, 5)}, ignore_index = True)
                current_df.to_csv(fullname, sep=';', index=False)
        
        fig_widget = scatter_plot('KNN', self.test_acc, self.dataset_name)
        self.container.children = tuple(list(self.container.children)[:self.plotting_level] +
                                            fig_widget) 
        self.container.children = tuple(list(self.container.children)[:self.plotting_level+1] +
                                            [self.go_to_visualisation_button]) 
        self.from_training = True
        self.go_to_visualisation_button.on_click(self.show_visualisation)

    def reg_rf(self, run):
        if self.current_algo == 'reg_rf_def':
            filename = 'reg_rf.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                current_df = pd.read_csv(fullname ,sep=';')

            else:
                current_df = pd.DataFrame(columns = ['n_estimators', 'criterion', 'max_depth', 'min_samples_split',
                                                        'min_samples_leaf','min_weight_fraction_leaf', 'max_features',
                                                        'max_leaf_nodes', 'min_impurity_decrease', 'bootstrap',
                                                        'oob_score', 'warm_start', 'ccp_alpha', 'max_samples', 'n_jobs',
                                                        'random_state', 'Mean CV train neg-MSE', 'Mean CV test neg-MSE',
                                                        'Mean CV fit time'])
            current_df['max_depth'] = current_df['max_depth'].astype('str')
            current_df['max_features'] = current_df['max_features'].astype('str')
            current_df['max_leaf_nodes'] = current_df['max_leaf_nodes'].astype('str')
            current_df['max_samples'] = current_df['max_samples'].astype('str')

            trained_before_df = current_df.loc[(current_df['n_estimators']==100)&
                                                             (current_df['criterion']=='mse')&
                                                             (current_df['max_depth']=='None')&
                                                             (current_df['min_samples_split']==2.0)&
                                                             (current_df['min_samples_leaf']==1.0)&
                                                             (current_df['min_weight_fraction_leaf']==0.0)&
                                                             (current_df['max_features']=='auto')&
                                                             (current_df['max_leaf_nodes']=='None')&
                                                             (current_df['min_impurity_decrease']==0.0)&
                                                             (current_df['bootstrap']==True)&
                                                             (current_df['oob_score']==False)&
                                                             (current_df['warm_start']==False)&
                                                             (current_df['ccp_alpha']==0.0)&
                                                             (current_df['max_samples']=='None')]
            if len(trained_before_df) > 0:
                result = widgets.Output()
                with result:
                    display(trained_before_df)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_current_df = widgets.HBox([result], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                            [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_current_df])
                self.test_acc = trained_before_df['Mean CV test neg-MSE'].values[0]
            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                clf = RandomForestRegressor()
                scores = cross_validate(clf, self.X_train, self.y_train, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
                train_acc = scores['train_score'].mean()
                self.test_acc = scores['test_score'].mean()
                fit_time = scores['fit_time'].mean()
                
                if len(current_df) > 0:
                    if self.test_acc > current_df['Mean CV train neg-MSE'].max():
                        result = f'Mean CV train neg-MSE: {round(train_acc,5)}. <br/>Mean CV test neg-MSE:  {round(self.test_acc,5)}. <br/>Best mean CV test neg-MSE for RF so far!'
                    else:
                        result = f'Mean CV train neg-MSE: {round(train_acc,5)}. <br/>Mean CV test neg-MSE:  {round(self.test_acc,5)}.'
                else:
                    result = f'Mean CV train neg-MSE: {round(train_acc,5)}. <br/>Mean CV test neg-MSE:  {round(self.test_acc,5)}. <br/>First RF model trained!'
                    
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is: <br/>'+ result)])
                current_df = current_df.append({'n_estimators': 100, 'criterion': 'mse',
                                                      'max_depth': str(None), 'min_samples_split': 2.0,
                                                      'min_samples_leaf': 1.0, 'min_weight_fraction_leaf': 0.0,
                                                      'max_features': 'auto', 'max_leaf_nodes': str(None),
                                                      'min_impurity_decrease': 0.0, 'bootstrap': True,
                                                      'oob_score': False, 'warm_start': False,'ccp_alpha': 0.0,
                                                      'max_samples': str(None), 'n_jobs': str(None), 'random_state': str(None),
                                                      'Mean CV train neg-MSE': round(train_acc, 5),
                                                      'Mean CV test neg-MSE': round(self.test_acc, 5),
                                                      'Mean CV fit time': round(fit_time, 5)}, ignore_index = True)
                current_df.to_csv(fullname, sep=';', index=False)

        elif self.current_algo == 'reg_rf_sup':
            filename = 'reg_rf.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                current_df = pd.read_csv(fullname ,sep=';')

            else:
                current_df = pd.DataFrame(columns = ['n_estimators', 'criterion', 'max_depth', 'min_samples_split',
                                                        'min_samples_leaf','min_weight_fraction_leaf', 'max_features',
                                                        'max_leaf_nodes', 'min_impurity_decrease', 'bootstrap',
                                                        'oob_score', 'warm_start', 'ccp_alpha', 'max_samples', 'n_jobs',
                                                        'random_state', 'Mean CV train neg-MSE', 'Mean CV test neg-MSE',
                                                        'Mean CV fit time'])

            current_df['max_depth'] = current_df['max_depth'].astype('str')
            current_df['max_features'] = current_df['max_features'].astype('str')

            n_estimators = self.current_hp_box.children[0].value
            max_depth = self.current_hp_box.children[1].children[0].value
            max_features = self.current_hp_box.children[2].children[0].value

            trained_before_df = current_df.loc[(current_df['n_estimators']==n_estimators)&
                                                             (current_df['max_depth']==str(max_depth))&
                                                             (current_df['max_features']==str(max_features))]
            if len(trained_before_df) > 0:
                result = widgets.Output()
                with result:
                    display(trained_before_df)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_current_df = widgets.HBox([result], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_current_df])
                self.test_acc = trained_before_df['Mean CV test neg-MSE'].values[0]
            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                try:
                    clf = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth,
                                                           max_features=max_features)
                    scores = cross_validate(clf, self.X_train, self.y_train, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
                    train_acc = scores['train_score'].mean()
                    self.test_acc = scores['test_score'].mean()
                    fit_time = scores['fit_time'].mean()
                
                except ValueError as err:
                    self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                    [widgets.HTML(str(err))])
                    return

                if len(current_df) > 0:
                    if self.test_acc > current_df['Mean CV train neg-MSE'].max():
                        result = f'Mean CV train neg-MSE: {round(train_acc,5)}. <br/>Mean CV test neg-MSE:  {round(self.test_acc,5)}. <br/>Best mean CV test neg-MSE for RF so far!'
                    else:
                        result = f'Mean CV train neg-MSE: {round(train_acc,5)}. <br/>Mean CV test neg-MSE:  {round(self.test_acc,5)}.'
                else:
                    result = f'Mean CV train neg-MSE: {round(train_acc,5)}. <br/>Mean CV test neg-MSE:  {round(self.test_acc,5)}. <br/>First RF model trained!'
                    
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is: <br/>'+ result)])

                current_df = current_df.append({'n_estimators': n_estimators, 'criterion': 'mse',
                                                      'max_depth': str(max_depth), 'min_samples_split': 2.0,
                                                      'min_samples_leaf': 1.0, 'min_weight_fraction_leaf': 0.0,
                                                      'max_features': str(max_features), 'max_leaf_nodes': str(None),
                                                      'min_impurity_decrease': 0.0, 'bootstrap': True,
                                                      'oob_score': False, 'warm_start': False,'ccp_alpha': 0.0,
                                                      'max_samples': str(None), 'n_jobs': str(None), 'random_state': str(None),
                                                      'Mean CV train neg-MSE': round(train_acc, 5),
                                                      'Mean CV test neg-MSE': round(self.test_acc, 5),
                                                      'Mean CV fit time': round(fit_time, 5)}, ignore_index = True)

                current_df['max_depth'] = current_df['max_depth'].astype('str')
                current_df['max_features'] = current_df['max_features'].astype('str')
                current_df['max_leaf_nodes'] = current_df['max_leaf_nodes'].astype('str')
                current_df['max_samples'] = current_df['max_samples'].astype('str')

                current_df.to_csv(fullname, sep=';', index=False)

        elif self.current_algo == 'reg_rf_pro':
            filename = 'reg_rf.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                current_df = pd.read_csv(fullname ,sep=';')

            else:
                current_df = pd.DataFrame(columns = ['n_estimators', 'criterion', 'max_depth', 'min_samples_split',
                                                        'min_samples_leaf','min_weight_fraction_leaf', 'max_features',
                                                        'max_leaf_nodes', 'min_impurity_decrease', 'bootstrap',
                                                        'oob_score', 'warm_start', 'ccp_alpha', 'max_samples', 'n_jobs',
                                                        'random_state', 'Mean CV train neg-MSE', 'Mean CV test neg-MSE',
                                                        'Mean CV fit time'])

            current_df['max_depth'] = current_df['max_depth'].astype('str')
            current_df['max_features'] = current_df['max_features'].astype('str')
            current_df['max_leaf_nodes'] = current_df['max_leaf_nodes'].astype('str')
            current_df['max_samples'] = current_df['max_samples'].astype('str')

            n_estimators = self.current_hp_box.children[0].children[0].children[0].children[1].value
            criterion = self.current_hp_box.children[0].children[1].children[0].children[1].value
            try:
                max_depth= int(self.current_hp_box.children[0].children[0].children[1].children[1].value)
            except ValueError:
                max_depth=None

            if self.current_hp_box.children[0].children[0].children[2].children[1].value <= 1:
                min_samples_split = float(self.current_hp_box.children[0].children[0].children[2].children[1].value)
            elif self.current_hp_box.children[0].children[0].children[2].children[1].value > 1:
                min_samples_split = int(self.current_hp_box.children[0].children[0].children[2].children[1].value)

            if self.current_hp_box.children[0].children[1].children[1].children[1].value <= 0.5:
                min_samples_leaf = float(self.current_hp_box.children[0].children[1].children[1].children[1].value)
            else:
                min_samples_leaf = int(self.current_hp_box.children[0].children[1].children[1].children[1].value)
            min_weight_fraction_leaf = self.current_hp_box.children[0].children[0].children[3].children[1].value
            try:
                max_features = float(self.current_hp_box.children[0].children[1].children[2].children[1].value)
                if max_features > 0:
                    max_features = int(max_features)
            except ValueError:
                if self.current_hp_box.children[0].children[1].children[2].children[1].value == 'None':
                    max_features=None
                else:
                    max_features = self.current_hp_box.children[0].children[1].children[2].children[1].value
            try:
                max_leaf_nodes = int(self.current_hp_box.children[0].children[0].children[4].children[1].value)
            except ValueError:
                max_leaf_nodes = None
            min_impurity_decrease=self.current_hp_box.children[0].children[1].children[3].children[1].value
            bootstrap = self.current_hp_box.children[0].children[0].children[5].children[1].value
            oob_score=self.current_hp_box.children[0].children[1].children[4].children[1].value
            n_jobs=self.current_hp_box.children[0].children[0].children[6].children[1].value
            verbose=self.current_hp_box.children[0].children[1].children[5].children[1].value
            warm_start=self.current_hp_box.children[0].children[0].children[7].children[1].value
            ccp_alpha=self.current_hp_box.children[0].children[1].children[6].children[1].value
            try:
                max_samples=float(self.current_hp_box.children[0].children[0].children[8].children[1].value)
            except ValueError:
                max_samples = None
            try:
                random_state = int(self.current_hp_box.children[0].children[1].children[7].children[1].value)
            except:
                random_state = None

            trained_before_df = current_df.loc[(current_df['n_estimators']==n_estimators)&
                                                             (current_df['criterion']==criterion)&
                                                             (current_df['max_depth']==str(max_depth))&
                                                             (current_df['min_samples_split']==min_samples_split)&
                                                             (current_df['min_samples_leaf']==min_samples_leaf)&
                                                             (current_df['min_weight_fraction_leaf']==min_weight_fraction_leaf)&
                                                             (current_df['max_features']==str(max_features))&
                                                             (current_df['max_leaf_nodes']==str(max_leaf_nodes))&
                                                             (current_df['min_impurity_decrease']==min_impurity_decrease)&
                                                             (current_df['bootstrap']==bootstrap)&
                                                             (current_df['oob_score']==oob_score)&
                                                             (current_df['warm_start']==warm_start)&
                                                             (current_df['ccp_alpha']==ccp_alpha)&
                                                             (current_df['max_samples']==str(max_samples))]
            if len(trained_before_df) > 0:
                
                result = widgets.Output()
                with result:
                    display(trained_before_df)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_current_df = widgets.HBox([result], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_current_df])
                
                self.test_acc = trained_before_df['Mean CV test neg-MSE'].values[0]

            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                try:
                    clf = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                                           min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                                           min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                                           max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
                                                           bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, verbose=verbose,
                                                           warm_start=warm_start, ccp_alpha=ccp_alpha, max_samples=max_samples,
                                                           random_state=random_state)
                    scores_reg_rf_pro = cross_validate(clf, self.X_train, self.y_train, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
                    train_acc = scores_reg_rf_pro['train_score'].mean()
                    self.test_acc = scores_reg_rf_pro['test_score'].mean()
                    fit_time = scores_reg_rf_pro['fit_time'].mean()
                except ValueError or ZeroDivisionError as err:
                    self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                    [widgets.HTML(str(err))])
                    return

                if len(current_df) > 0:
                    if self.test_acc > current_df['Mean CV train neg-MSE'].max():
                        result = f'Mean CV train neg-MSE: {round(train_acc,5)}. <br/>Mean CV test neg-MSE:  {round(self.test_acc,5)}. <br/>Best mean CV test neg-MSE for RF so far!'
                    else:
                        result = f'Mean CV train neg-MSE: {round(train_acc,5)}. <br/>Mean CV test neg-MSE:  {round(self.test_acc,5)}.'
                else:
                    result = f'Mean CV train neg-MSE: {round(train_acc,5)}. <br/>Mean CV test neg-MSE:  {round(self.test_acc,5)}. <br/>First RF model trained!'
                    
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is: <br/>'+ result)])

                current_df = current_df.append({'n_estimators': n_estimators, 'criterion': criterion,
                                                      'max_depth': str(max_depth), 'min_samples_split': min_samples_split,
                                                      'min_samples_leaf': min_samples_leaf, 'min_weight_fraction_leaf': min_weight_fraction_leaf,
                                                      'max_features': str(max_features), 'max_leaf_nodes': str(max_leaf_nodes),
                                                      'min_impurity_decrease': min_impurity_decrease, 'bootstrap': bootstrap,
                                                      'oob_score': oob_score, 'warm_start': warm_start,'ccp_alpha': ccp_alpha,
                                                      'max_samples': str(max_samples), 'n_jobs': str(n_jobs), 'random_state': str(random_state),
                                                      'Mean CV train neg-MSE': round(train_acc, 5),
                                                      'Mean CV test neg-MSE': round(self.test_acc, 5),
                                                      'Mean CV fit time': round(fit_time, 5)}, ignore_index = True)
                current_df.to_csv(fullname, sep=';', index=False)

        fig_widget = scatter_plot('Random Forest', self.test_acc, self.dataset_name, False)
        self.container.children = tuple(list(self.container.children)[:self.plotting_level] +
                                            fig_widget) 
        self.container.children = tuple(list(self.container.children)[:self.plotting_level+1] +
                                            [self.go_to_visualisation_button]) 
        self.from_training = True
        self.go_to_visualisation_button.on_click(self.show_visualisation)

    def class_rf(self, run):
        if self.current_algo == 'class_rf_def':
            filename = 'class_rf.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                current_df = pd.read_csv(fullname ,sep=';')

            else:
                current_df = pd.DataFrame(columns = ['n_estimators', 'criterion', 'max_depth', 'min_samples_split',
                                                        'min_samples_leaf','min_weight_fraction_leaf', 'max_features',
                                                        'max_leaf_nodes', 'min_impurity_decrease', 'bootstrap',
                                                        'oob_score', 'warm_start', 'class_weight', 'ccp_alpha', 'max_samples',
                                                        'n_jobs', 'random_state', 'Mean CV train acc', 'Mean CV test acc',
                                                          'Mean CV fit time'])

            current_df['max_depth'] = current_df['max_depth'].astype('str')
            current_df['max_features'] = current_df['max_features'].astype('str')
            current_df['max_leaf_nodes'] = current_df['max_leaf_nodes'].astype('str')
            current_df['max_samples'] = current_df['max_samples'].astype('str')
            current_df['class_weight'] = current_df['class_weight'].astype('str')

            trained_before_df = current_df.loc[(current_df['n_estimators']==100)&
                                                             (current_df['criterion']=='gini')&
                                                             (current_df['max_depth']=='None')&
                                                             (current_df['min_samples_split']==2.0)&
                                                             (current_df['min_samples_leaf']==1.0)&
                                                             (current_df['min_weight_fraction_leaf']==0.0)&
                                                             (current_df['max_features']=='auto')&
                                                             (current_df['max_leaf_nodes']=='None')&
                                                             (current_df['min_impurity_decrease']==0.0)&
                                                             (current_df['bootstrap']==True)&
                                                             (current_df['oob_score']==False)&
                                                             (current_df['warm_start']==False)&
                                                             (current_df['class_weight']=='None')&
                                                             (current_df['ccp_alpha']==0.0)&
                                                             (current_df['max_samples']=='None')]
            if len(trained_before_df) > 0:
                result = widgets.Output()
                with result:
                    display(trained_before_df)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_current_df = widgets.HBox([result], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_current_df])
                self.test_acc = trained_before_df['Mean CV test acc'].values[0]

            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                clf = RandomForestClassifier()
                scores = cross_validate(clf, self.X_train, self.y_train, cv=5, scoring='balanced_accuracy', return_train_score=True)
                train_acc = scores['train_score'].mean()
                self.test_acc = scores['test_score'].mean()
                fit_time = scores['fit_time'].mean()
                
                if len(current_df) > 0:
                    if self.test_acc > current_df['Mean CV test acc'].max():
                        result = f'Mean CV train accuracy: {round(train_acc,2)*100}%. <br/>Mean CV test accuracy:  {round(self.test_acc,2)*100}%. <br/>Best mean CV test accuracy for RF so far!'
                    else:
                        result = f'Mean CV train accuracy: {round(train_acc,2)*100}%. <br/>Mean CV test accuracy:  {round(self.test_acc,2)*100}%.'
                else:
                    result = f'Mean CV train accuracy: {round(train_acc,2)*100}%. <br/>Mean CV test accuracy:  {round(self.test_acc,2)*100}%. <br/>First RF model trained!'
                    
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is:  <br/>'+ result)])
                current_df = current_df.append({'n_estimators': 100, 'criterion': 'gini',
                                                      'max_depth': str(None), 'min_samples_split': 2.0,
                                                      'min_samples_leaf': 1.0, 'min_weight_fraction_leaf': 0.0,
                                                      'max_features': 'auto', 'max_leaf_nodes': str(None),
                                                      'min_impurity_decrease': 0.0, 'bootstrap': True,
                                                      'oob_score': False, 'warm_start': False,'class_weight': str(None), 'ccp_alpha': 0.0,
                                                      'max_samples': str(None), 'n_jobs': str(None), 'random_state': str(None),
                                                      'Mean CV train acc': round(train_acc, 5),
                                                      'Mean CV test acc': round(self.test_acc, 5),
                                                      'Mean CV fit time': round(fit_time, 5)}, ignore_index = True)
                current_df.to_csv(fullname, sep=';', index=False)

        elif self.current_algo == 'class_rf_sup':
            filename = 'class_rf.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                current_df = pd.read_csv(fullname ,sep=';')

            else:
                current_df = pd.DataFrame(columns = ['n_estimators', 'criterion', 'max_depth', 'min_samples_split',
                                                          'min_samples_leaf','min_weight_fraction_leaf', 'max_features',
                                                          'max_leaf_nodes', 'min_impurity_decrease', 'bootstrap',
                                                          'oob_score', 'warm_start', 'class_weight', 'ccp_alpha', 'max_samples',
                                                          'n_jobs', 'random_state', 'Mean CV train acc', 'Mean CV test acc',
                                                          'Mean CV fit time'])

            n_estimators = self.current_hp_box.children[0].value
            max_depth = self.current_hp_box.children[1].children[0].value
            max_features = self.current_hp_box.children[2].children[0].value

            current_df['max_depth'] = current_df['max_depth'].astype('str')
            current_df['max_features'] = current_df['max_features'].astype('str')

            trained_before_df = current_df.loc[(current_df['n_estimators']==n_estimators)&
                                                             (current_df['max_depth']==str(max_depth))&
                                                             (current_df['max_features']==str(max_features))]
            if len(trained_before_df) > 0:
                result = widgets.Output()
                with result:
                    display(trained_before_df)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_current_df = widgets.HBox([result], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_current_df])
                self.test_acc = trained_before_df['Mean CV test acc'].values[0]
            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                try:
                    clf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,
                                                           max_features=max_features)
                    scores = cross_validate(clf, self.X_train, self.y_train, cv=5, scoring='balanced_accuracy', return_train_score=True)
                    train_acc = scores['train_score'].mean()
                    self.test_acc = scores['test_score'].mean()
                    fit_time = scores['fit_time'].mean()
                
                except ValueError as err:
                    self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                    [widgets.HTML(str(err))])
                    return

                if len(current_df) > 0:
                    if self.test_acc > current_df['Mean CV test acc'].max():
                        result = f'Mean CV train accuracy: {round(train_acc,2)*100}%. <br/>Mean CV test accuracy:  {round(self.test_acc,2)*100}%. <br/>Best mean CV test accuracy for RF so far!'
                    else:
                        result = f'Mean CV train accuracy: {round(train_acc,2)*100}%. <br/>Mean CV test accuracy:  {round(self.test_acc,2)*100}%.'
                else:
                    result = f'Mean CV train accuracy: {round(train_acc,2)*100}%. <br/>Mean CV test accuracy:  {round(self.test_acc,2)*100}%. <br/>First RF model trained!'
                    
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is: <br/>'+ result)])

                current_df = current_df.append({'n_estimators': n_estimators, 'criterion': 'gini',
                                                          'max_depth': str(max_depth), 'min_samples_split': 2.0,
                                                          'min_samples_leaf': 1.0, 'min_weight_fraction_leaf': 0.0,
                                                          'max_features': str(max_features), 'max_leaf_nodes': str(None),
                                                          'min_impurity_decrease': 0.0, 'bootstrap': True,
                                                          'oob_score': False, 'warm_start': False,'class_weight': str(None), 'ccp_alpha': 0.0,
                                                          'max_samples': str(None), 'n_jobs': str(None), 'random_state': str(None),
                                                          'Mean CV train acc': round(train_acc, 5),
                                                          'Mean CV test acc': round(self.test_acc, 5),
                                                          'Mean CV fit time': round(fit_time, 5)}, ignore_index = True)

                current_df.to_csv(fullname, sep=';', index=False)

        elif self.current_algo == 'class_rf_pro':
            filename = 'class_rf.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                current_df = pd.read_csv(fullname ,sep=';')

            else:
                current_df = pd.DataFrame(columns = ['n_estimators', 'criterion', 'max_depth', 'min_samples_split',
                                                          'min_samples_leaf','min_weight_fraction_leaf', 'max_features',
                                                          'max_leaf_nodes', 'min_impurity_decrease', 'bootstrap',
                                                          'oob_score', 'warm_start', 'class_weight', 'ccp_alpha', 'max_samples',
                                                          'n_jobs', 'random_state', 'Mean CV train acc', 'Mean CV test acc',
                                                          'Mean CV fit time'])


            current_df['max_depth'] = current_df['max_depth'].astype('str')
            current_df['max_features'] = current_df['max_features'].astype('str')
            current_df['max_leaf_nodes'] = current_df['max_leaf_nodes'].astype('str')
            current_df['max_samples'] = current_df['max_samples'].astype('str')
            current_df['class_weight'] = current_df['class_weight'].astype('str')

            n_estimators = self.current_hp_box.children[0].children[0].children[0].children[1].value
            criterion = self.current_hp_box.children[0].children[1].children[0].children[1].value
            try:
                max_depth=int(self.current_hp_box.children[0].children[0].children[1].children[1].value)
            except ValueError:
                max_depth=None

            if self.current_hp_box.children[0].children[1].children[1].children[1].value <= 1:
                min_samples_split = float(self.current_hp_box.children[0].children[1].children[1].children[1].value)
            elif self.current_hp_box.children[0].children[1].children[1].children[1].value > 1:
                min_samples_split = int(self.current_hp_box.children[0].children[1].children[1].children[1].value)

            if self.current_hp_box.children[0].children[0].children[2].children[1].value <= 0.5:
                min_samples_leaf = float(self.current_hp_box.children[0].children[0].children[2].children[1].value)
            else:
                min_samples_leaf = int(self.current_hp_box.children[0].children[0].children[2].children[1].value)

            min_weight_fraction_leaf = self.current_hp_box.children[0].children[1].children[2].children[1].value
            try:
                max_features = float(self.current_hp_box.children[0].children[0].children[3].children[1].value)
            except ValueError:
                if self.current_hp_box.children[0].children[0].children[3].children[1].value == 'None':
                    max_features=None
                else:
                    max_features = self.current_hp_box.children[0].children[0].children[3].children[1].value
            try:
                max_leaf_nodes = int(self.current_hp_box.children[0].children[1].children[3].children[1].value)
            except ValueError:
                max_leaf_nodes = None
            min_impurity_decrease = self.current_hp_box.children[0].children[0].children[4].children[1].value
            bootstrap = self.current_hp_box.children[0].children[1].children[4].children[1].value
            oob_score=self.current_hp_box.children[0].children[0].children[5].children[1].value
            n_jobs=self.current_hp_box.children[0].children[1].children[5].children[1].value
            verbose= self.current_hp_box.children[0].children[0].children[6].children[1].value
            warm_start=self.current_hp_box.children[0].children[1].children[6].children[1].value
            class_weight = self.current_hp_box.children[0].children[0].children[7].children[1].value
            ccp_alpha= self.current_hp_box.children[0].children[1].children[7].children[1].value
            try:
                max_samples= float(self.current_hp_box.children[0].children[0].children[8].children[1].value)
            except ValueError:
                max_samples = None
            try:
                random_state = int(self.current_hp_box.children[0].children[1].children[8].children[1].value)
            except:
                random_state = None

            trained_before_df = current_df.loc[(current_df['n_estimators']==n_estimators)&
                                                                 (current_df['criterion']==criterion)&
                                                                 (current_df['max_depth']==str(max_depth))&
                                                                 (current_df['min_samples_split']==min_samples_split)&
                                                                 (current_df['min_samples_leaf']==min_samples_leaf)&
                                                                 (current_df['min_weight_fraction_leaf']==min_weight_fraction_leaf)&
                                                                 (current_df['max_features']==str(max_features))&
                                                                 (current_df['max_leaf_nodes']==str(max_leaf_nodes))&
                                                                 (current_df['min_impurity_decrease']==min_impurity_decrease)&
                                                                 (current_df['bootstrap']==bootstrap)&
                                                                 (current_df['oob_score']==oob_score)&
                                                                 (current_df['warm_start']==warm_start)&
                                                                 (current_df['class_weight']==str(class_weight))&
                                                                 (current_df['ccp_alpha']==ccp_alpha)&
                                                                 (current_df['max_samples']==str(max_samples))]
            if len(trained_before_df) > 0:
                result = widgets.Output()
                with result:
                    display(trained_before_df)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_current_df = widgets.HBox([result], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_current_df])
                self.test_acc = trained_before_df['Mean CV test acc'].values[0]
            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                try:
                    clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                                           min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                                           min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                                           max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
                                                           bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, verbose=verbose,
                                                           warm_start=warm_start, class_weight=class_weight, ccp_alpha=ccp_alpha,
                                                           max_samples=max_samples, random_state=random_state)
                    scores = cross_validate(clf, self.X_train, self.y_train, cv=5, scoring='balanced_accuracy', return_train_score=True)
                    train_acc = scores['train_score'].mean()
                    self.test_acc = scores['test_score'].mean()
                    fit_time = scores['fit_time'].mean()
                
                except ValueError as err:
                    self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                    [widgets.HTML(str(err))])
                    return

                if len(current_df) > 0:
                    if self.test_acc > current_df['Mean CV test acc'].max():
                        result = f'Mean CV train accuracy: {round(train_acc,2)*100}%. <br/>Mean CV test accuracy:  {round(self.test_acc,2)*100}%. <br/>Best mean CV test accuracy for RF so far!'
                    else:
                        result = f'Mean CV train accuracy: {round(train_acc,2)*100}%. <br/>Mean CV test accuracy:  {round(self.test_acc,2)*100}%.'
                else:
                    result = f'Mean CV train accuracy: {round(train_acc,2)*100}%. <br/>Mean CV test accuracy:  {round(self.test_acc,2)*100}%. <br/>First RF model trained!'
                    
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is: <br/>'+ result)])

                current_df = current_df.append({'n_estimators': n_estimators, 'criterion': criterion,
                                                          'max_depth': str(max_depth), 'min_samples_split': min_samples_split,
                                                          'min_samples_leaf': min_samples_leaf, 'min_weight_fraction_leaf': min_weight_fraction_leaf,
                                                          'max_features': str(max_features), 'max_leaf_nodes': str(max_leaf_nodes),
                                                          'min_impurity_decrease': min_impurity_decrease, 'bootstrap': bootstrap,
                                                          'oob_score': oob_score, 'warm_start': warm_start,'class_weight': str(class_weight), 'ccp_alpha': ccp_alpha,
                                                          'max_samples': str(max_samples), 'n_jobs': str(n_jobs), 'random_state': str(random_state),
                                                          'Mean CV train acc': round(train_acc, 5),
                                                          'Mean CV test acc': round(self.test_acc, 5),
                                                          'Mean CV fit time': round(fit_time, 5)}, ignore_index = True)
                current_df.to_csv(fullname, sep=';', index=False)
        
        fig_widget = scatter_plot('Random Forest', self.test_acc, self.dataset_name)
        self.container.children = tuple(list(self.container.children)[:self.plotting_level] +
                                            fig_widget) 
        self.container.children = tuple(list(self.container.children)[:self.plotting_level+1] +
                                            [self.go_to_visualisation_button]) 
        self.from_training = True
        self.go_to_visualisation_button.on_click(self.show_visualisation)

    def class_svm(self, run):
        if self.current_algo == 'class_svm_def':
            filename = 'class_svm.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                current_df = pd.read_csv(fullname ,sep=';')

            else:
                current_df = pd.DataFrame(columns = ['C', 'kernel', 'degree', 'gamma', 'coef0', 'shrinking',
                                                           'probability', 'tol', 'cache_size', 'class_weight',
                                                           'max_iter', 'decision_function_shape',
                                                           'break_ties', 'random_state', 'Mean CV train acc', 'Mean CV test acc',
                                                          'Mean CV fit time'])

            trained_before_df = current_df.loc[(current_df['C']==1.0)&
                                                                   (current_df['kernel']=='rbf')&
                                                                   (current_df['degree']==3)&
                                                                   (current_df['gamma']=='scale')&
                                                                   (current_df['coef0']==0.0)&
                                                                   (current_df['shrinking']==True)&
                                                                   (current_df['probability']==False)&
                                                                   (current_df['tol']==0.001)&
                                                                   (current_df['cache_size']==200.0)&
                                                                   (current_df['class_weight']=='None')&
                                                                   (current_df['max_iter']==-1)&
                                                                   (current_df['decision_function_shape']=='ovr')&
                                                                   (current_df['break_ties']==False)]
            if len(trained_before_df) > 0:
                result = widgets.Output()
                with result:
                    display(trained_before_df)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_current_df = widgets.HBox([result], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_current_df])
                self.test_acc = trained_before_df['Mean CV test acc'].values[0]
            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                clf = SVC()
                scores = cross_validate(clf, self.X_train, self.y_train, cv=5, scoring='balanced_accuracy', return_train_score=True)
                train_acc = scores['train_score'].mean()
                self.test_acc = scores['test_score'].mean()
                fit_time = scores['fit_time'].mean()

                if len(current_df) > 0:
                    if self.test_acc > current_df['Mean CV test acc'].max():
                        result = f'Mean CV train accuracy: {round(train_acc,2)*100}%. <br/>Mean CV test accuracy:  {round(self.test_acc,2)*100}%. <br/>Best mean CV test accuracy for SVM so far!'
                    else:
                        result = f'Mean CV train accuracy: {round(train_acc,2)*100}%. <br/>Mean CV test accuracy:  {round(self.test_acc,2)*100}%.'
                else:
                    result = f'Mean CV train accuracy: {round(train_acc,2)*100}%. <br/>Mean CV test accuracy:  {round(self.test_acc,2)*100}%. <br/>First SVM model trained!'
                    
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is:  <br/>'+ result)])
                current_df = current_df.append({'C': 1.0, 'kernel': 'rbf', 'degree': 3, 'gamma': 'scale',
                                                           'coef0':0.0, 'shrinking': True, 'probability': False,
                                                           'tol': 0.001, 'cache_size': 200.0, 'class_weight': str(None),
                                                           'max_iter': -1, 'decision_function_shape': 'ovr',
                                                           'break_ties': False, 'random_state': str(None),
                                                           'Mean CV train acc': round(train_acc, 5),
                                                           'Mean CV test acc': round(self.test_acc, 5),
                                                           'Mean CV fit time': round(fit_time, 5)},
                                                          ignore_index = True)
                current_df.to_csv(fullname, sep=';', index=False)

        elif self.current_algo == 'class_svm_sup':
            filename = 'class_svm.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                current_df = pd.read_csv(fullname ,sep=';')

            else:
                current_df = pd.DataFrame(columns = ['C', 'kernel', 'degree', 'gamma', 'coef0', 'shrinking',
                                                           'probability', 'tol', 'cache_size', 'class_weight',
                                                           'max_iter', 'decision_function_shape',
                                                           'break_ties', 'random_state', 'Mean CV train acc', 'Mean CV test acc',
                                                          'Mean CV fit time'])

            c = self.current_hp_box.children[0].value
            kernel = self.current_hp_box.children[1].value
            gamma = self.current_hp_box.children[2].value

            trained_before_df = current_df.loc[(current_df['C']==c)&
                                                                   (current_df['kernel']==kernel)&
                                                                   (current_df['gamma']==str(gamma))]
            if len(trained_before_df) > 0:
                result = widgets.Output()
                with result:
                    display(trained_before_df)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_current_df = widgets.HBox([result], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_current_df])
                self.test_acc = trained_before_df['Mean CV test acc'].values[0]
            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                try:
                    clf = SVC(C=c, kernel=kernel, gamma=gamma)
                    scores = cross_validate(clf, self.X_train, self.y_train, cv=5, scoring='balanced_accuracy', return_train_score=True)
                    train_acc = scores['train_score'].mean()
                    self.test_acc = scores['test_score'].mean()
                    fit_time = scores['fit_time'].mean()
                    
                except ValueError as err:
                    self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                    [widgets.HTML(str(err))])
                    return

                if len(current_df) > 0:
                    if self.test_acc > current_df['Mean CV test acc'].max():
                        result = f'Mean CV train accuracy: {round(train_acc,2)*100}%. <br/>Mean CV test accuracy:  {round(self.test_acc,2)*100}%. <br/>Best mean CV test accuracy for SVM so far!'
                    else:
                        result = f'Mean CV train accuracy: {round(train_acc,2)*100}%. <br/>Mean CV test accuracy:  {round(self.test_acc,2)*100}%.'
                else:
                    result = f'Mean CV train accuracy: {round(train_acc,2)*100}%. <br/>Mean CV test accuracy:  {round(self.test_acc,2)*100}%. <br/>First SVM model trained!'
                    
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is:  <br/>'+ result)])
                current_df = current_df.append({'C': c, 'kernel': kernel, 'degree': 3, 'gamma': str(gamma),
                                                            'coef0':0.0, 'shrinking': True, 'probability': False,
                                                            'tol': 0.001, 'cache_size': 200.0, 'class_weight': str(None),
                                                            'max_iter': -1, 'decision_function_shape': 'ovr',
                                                            'break_ties': False, 'random_state': str(None),
                                                            'Mean CV train acc': round(train_acc, 5),
                                                            'Mean CV test acc': round(self.test_acc, 5),
                                                            'Mean CV fit time': round(fit_time, 5)},
                                                           ignore_index = True)
                current_df.to_csv(fullname, sep=';', index=False)

        elif self.current_algo == 'class_svm_pro':
            filename = 'class_svm.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                current_df = pd.read_csv(fullname ,sep=';')

            else:
                current_df = pd.DataFrame(columns = ['C', 'kernel', 'degree', 'gamma', 'coef0', 'shrinking',
                                                           'probability', 'tol', 'cache_size', 'class_weight',
                                                           'max_iter', 'decision_function_shape',
                                                           'break_ties', 'random_state', 'Mean CV train acc', 'Mean CV test acc',
                                                          'Mean CV fit time'])

            c = self.current_hp_box.children[0].children[0].children[0].children[1].value
            kernel = self.current_hp_box.children[0].children[1].children[0].children[1].value
            degree = self.current_hp_box.children[0].children[0].children[1].children[1].value
            try:
                gamma = float(self.current_hp_box.children[0].children[0].children[2].children[1].value)
            except:
                gamma = self.current_hp_box.children[0].children[0].children[2].children[1].value
            coef0 = self.current_hp_box.children[0].children[0].children[3].children[1].value
            shrinking = self.current_hp_box.children[0].children[1].children[1].children[1].value
            probability = self.current_hp_box.children[0].children[0].children[4].children[1].value
            tol = self.current_hp_box.children[0].children[1].children[2].children[1].value
            cache_size = self.current_hp_box.children[0].children[0].children[5].children[1].value
            class_weight = self.current_hp_box.children[0].children[1].children[3].children[1].value
            max_iter = self.current_hp_box.children[0].children[1].children[4].children[1].value
            verbose = self.current_hp_box.children[0].children[0].children[6].children[1].value
            decision_function_shape = self.current_hp_box.children[0].children[0].children[7].children[1].value
            break_ties = self.current_hp_box.children[0].children[1].children[5].children[1].value
            try:
                random_state = int(self.current_hp_box.children[0].children[1].children[6].children[1].value)
            except:
                random_state = None

            trained_before_df = current_df.loc[(current_df['C']==c)&
                                                                   (current_df['kernel']==kernel)&
                                                                   (current_df['degree']==degree)&
                                                                   (current_df['gamma']==str(gamma))&
                                                                   (current_df['coef0']==coef0)&
                                                                   (current_df['shrinking']==shrinking)&
                                                                   (current_df['probability']==probability)&
                                                                   (current_df['tol']==tol)&
                                                                   (current_df['cache_size']==cache_size)&
                                                                   (current_df['class_weight']==str(class_weight))&
                                                                   (current_df['max_iter']==max_iter)&
                                                                   (current_df['decision_function_shape']==decision_function_shape)&
                                                                   (current_df['break_ties']==break_ties)]
            if len(trained_before_df) > 0:
                result = widgets.Output()
                with result:
                    display(trained_before_df)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_current_df = widgets.HBox([result], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_current_df])
                self.test_acc = trained_before_df['Mean CV test acc'].values[0]
            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                try:
                    clf = SVC(C=c, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0,
                                            probability=probability, shrinking=shrinking, tol=tol,
                                            cache_size=cache_size, class_weight=class_weight, verbose=verbose,
                                            decision_function_shape=decision_function_shape, max_iter=max_iter,
                                            break_ties=break_ties, random_state=random_state)
                    scores = cross_validate(clf, self.X_train, self.y_train, cv=5, scoring='balanced_accuracy', return_train_score=True)
                    train_acc = scores['train_score'].mean()
                    self.test_acc = scores['test_score'].mean()
                    fit_time = scores['fit_time'].mean()
                except ValueError as err:
                    self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                    [widgets.HTML(str(err))])
                    return

                if len(current_df) > 0:
                    if self.test_acc > current_df['Mean CV test acc'].max():
                        result = f'Mean CV train accuracy: {round(train_acc,2)*100}%. <br/>Mean CV test:  {round(self.test_acc,2)*100}%. <br/>Best mean CV test accuracy for SVM so far!'
                    else:
                        result = f'Mean CV train accuracy: {round(train_acc,2)*100}%. <br/>Mean CV test accuracy:  {round(self.test_acc,2)*100}%.'
                else:
                    result = f'Mean CV train accuracy: {round(train_acc,2)*100}%. <br/>Mean CV test accuracy:  {round(self.test_acc,2)*100}%. <br/>First SVM model trained!'
                    
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is:  <br/>'+ result)])
                current_df = current_df.append({'C': c, 'kernel': kernel, 'degree': degree, 'gamma': str(gamma),
                                                            'coef0': coef0, 'shrinking': shrinking, 'probability': probability,
                                                            'tol': tol, 'cache_size': cache_size, 'class_weight': str(class_weight),
                                                            'max_iter': max_iter, 'decision_function_shape': decision_function_shape,
                                                            'break_ties': break_ties,'random_state': str(random_state),
                                                            'Mean CV train acc': round(train_acc, 5),
                                                            'Mean CV test acc': round(self.test_acc, 5),
                                                            'Mean CV fit time': round(fit_time, 5)},
                                                           ignore_index = True)
                current_df.to_csv(fullname, sep=';', index=False)

        fig_widget = scatter_plot('SVM', self.test_acc, self.dataset_name)
        self.container.children = tuple(list(self.container.children)[:self.plotting_level] +
                                            fig_widget) 
        self.container.children = tuple(list(self.container.children)[:self.plotting_level+1] +
                                            [self.go_to_visualisation_button]) 
        self.from_training = True
        self.go_to_visualisation_button.on_click(self.show_visualisation)

    
    def reg_lin(self, run):
        if self.current_algo == 'reg_lin_def':
            filename = 'reg_lin.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                current_df = pd.read_csv(fullname ,sep=';')

            else:
                current_df = pd.DataFrame(columns = ['fit_intercept', 'normalize', 'copy_X', 'positive', 'n_jobs',
                                                         'Mean CV train neg-MSE', 'Mean CV test neg-MSE',
                                                          'Mean CV fit time'])

            trained_before_df = current_df.loc[(current_df['fit_intercept']==True)&
                                                               (current_df['normalize']==False)&
                                                               (current_df['copy_X']==True)&
                                                               (current_df['positive']==False)]
            if len(trained_before_df) > 0:
                result = widgets.Output()
                with result:
                    display(trained_before_df)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_current_df = widgets.HBox([result], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_current_df])
                self.test_acc = trained_before_df['Mean CV test neg-MSE'].values[0]
            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                clf = LinearRegression()
                scores = cross_validate(clf, self.X_train, self.y_train, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
                train_acc = scores['train_score'].mean()
                self.test_acc = scores['test_score'].mean()
                fit_time = scores['fit_time'].mean()

                if len(current_df) > 0:
                    if self.test_acc > current_df['Mean CV train neg-MSE'].max():
                        result = f'Mean CV train neg-MSE: {round(train_acc,5)}. <br/>Mean CV test neg-MSE:  {round(self.test_acc,5)}. <br/>Best mean CV test neg-MSE for Linear Regression so far!'
                    else:
                        result = f'Mean CV train neg-MSE: {round(train_acc,5)}. <br/>Mean CV test neg-MSE:  {round(self.test_acc,5)}.'
                else:
                    result = f'Mean CV train neg-MSE: {round(train_acc,5)}. <br/>Mean CV test neg-MSE:  {round(self.test_acc,5)}. <br/>First Linear Regression model trained!'
                    
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is:  \n'+ result)])
                current_df = current_df.append({'fit_intercept': True, 'normalize': False,
                                                       'copy_X': True, 'positive': False, 'n_jobs': str(None),
                                                       'Mean CV train neg-MSE': round(train_acc, 5),
                                                       'Mean CV test neg-MSE': round(self.test_acc, 5),
                                                       'Mean CV fit time': round(fit_time, 5)},
                                                      ignore_index = True)
                current_df.to_csv(fullname, sep=';', index=False)

        elif self.current_algo == 'reg_lin_sup':
            filename = 'reg_lin.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                current_df = pd.read_csv(fullname ,sep=';')

            else:
                current_df = pd.DataFrame(columns = ['fit_intercept', 'normalize', 'copy_X', 'positive', 'n_jobs',
                                                         'Mean CV train neg-MSE', 'Mean CV test neg-MSE',
                                                          'Mean CV fit time'])

            trained_before_df = current_df.loc[(current_df['fit_intercept']==True)&
                                                               (current_df['normalize']==False)&
                                                               (current_df['copy_X']==True)&
                                                               (current_df['positive']==False)]
            if len(trained_before_df) > 0:
                result = widgets.Output()
                with result:
                    display(trained_before_df)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_current_df = widgets.HBox([result], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_current_df])
                self.test_acc = trained_before_df['Mean CV test neg-MSE'].values[0]
            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                clf = LinearRegression()
                scores = cross_validate(clf, self.X_train, self.y_train, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
                train_acc = scores['train_score'].mean()
                self.test_acc = scores['test_score'].mean()
                fit_time = scores['fit_time'].mean()

                if len(current_df) > 0:
                    if self.test_acc > current_df['Mean CV train neg-MSE'].max():
                        result = f'Mean CV train neg-MSE: {round(train_acc,5)}. <br/>Mean CV test neg-MSE:  {round(self.test_acc,5)}. <br/>Best mean CV test neg-MSE for Linear Regression so far!'
                    else:
                        result = f'Mean CV train neg-MSE: {round(train_acc,5)}. <br/>Mean CV test neg-MSE:  {round(self.test_acc,5)}.'
                else:
                    result = f'Mean CV train neg-MSE: {round(train_acc,5)}. <br/>Mean CV test neg-MSE:  {round(self.test_acc,5)}. <br/>First Linear Regression model trained!'
                    
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is:  <br/>'+ result)])
                current_df = current_df.append({'fit_intercept': True, 'normalize': False,
                                                        'copy_X': True, 'positive': False, 'n_jobs': str(None),
                                                        'Mean CV train neg-MSE': round(train_acc, 5),
                                                        'Mean CV test neg-MSE': round(self.test_acc, 5),
                                                        'Mean CV fit time': round(fit_time, 5)},
                                                       ignore_index = True)
                current_df.to_csv(fullname, sep=';', index=False)
        
        elif self.current_algo == 'reg_lin_pro':
            filename = 'reg_lin.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                current_df = pd.read_csv(fullname ,sep=';')

            else:
                current_df = pd.DataFrame(columns = ['fit_intercept', 'normalize', 'copy_X', 'positive', 'n_jobs',
                                                         'Mean CV train neg-MSE', 'Mean CV test neg-MSE',
                                                          'Mean CV fit time'])

            fit_intercept=self.current_hp_box.children[0].children[0].children[0].children[1].value
            normalize=self.current_hp_box.children[0].children[1].children[0].children[1].value
            copy_X=self.current_hp_box.children[0].children[0].children[1].children[1].value
            n_jobs=self.current_hp_box.children[0].children[1].children[1].children[1].value
            positive = self.current_hp_box.children[0].children[0].children[2].children[1].value

            trained_before_df = current_df.loc[(current_df['fit_intercept']==fit_intercept)&
                                                               (current_df['normalize']==normalize)&
                                                               (current_df['copy_X']==copy_X)&
                                                               (current_df['positive']==positive)]
            if len(trained_before_df) > 0:
                result = widgets.Output()
                with result:
                    display(trained_before_df)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_current_df = widgets.HBox([result], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_current_df])
                self.test_acc = trained_before_df['Mean CV test neg-MSE'].values[0]
            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                try:
                    clf = LinearRegression(fit_intercept=fit_intercept, normalize=normalize,
                                                       copy_X=copy_X, n_jobs=n_jobs, positive=positive)
                    scores = cross_validate(clf, self.X_train, self.y_train, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
                    train_acc = scores['train_score'].mean()
                    self.test_acc = scores['test_score'].mean()
                    fit_time = scores['fit_time'].mean()

                except ValueError as err:
                    self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                    [widgets.HTML(str(err))])
                    return


                if len(current_df) > 0:
                    if self.test_acc > current_df['Mean CV train neg-MSE'].max():
                        result = f'Mean CV train neg-MSE: {round(train_acc,5)}. <br/>Mean CV test neg-MSE:  {round(self.test_acc,5)}. <br/>Best mean CV test neg-MSE for Linear Regression so far!'
                    else:
                        result = f'Mean CV train neg-MSE: {round(train_acc,5)}. <br/>Mean CV test neg-MSE:  {round(self.test_acc,5)}.'
                else:
                    result = f'Mean CV train neg-MSE: {round(train_acc,5)}. <br/>Mean CV test neg-MSE:  {round(self.test_acc,5)}. <br/>First Linear Regression model trained!'
                    
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is:  <br/>'+ result)])
                current_df = current_df.append({'fit_intercept': fit_intercept, 'normalize': normalize,
                                                        'copy_X': copy_X, 'positive': positive, 'n_jobs': str(n_jobs),
                                                        'Mean CV train neg-MSE': round(train_acc, 5),
                                                        'Mean CV test neg-MSE': round(self.test_acc, 5),
                                                        'Mean CV fit time': round(fit_time, 5)},
                                                       ignore_index = True)
                current_df.to_csv(fullname, sep=';', index=False)

        fig_widget = scatter_plot('Linear Regression', self.test_acc, self.dataset_name, False)
        self.container.children = tuple(list(self.container.children)[:self.plotting_level] +
                                            fig_widget) 
        self.container.children = tuple(list(self.container.children)[:self.plotting_level+1] +
                                            [self.go_to_visualisation_button]) 
        self.from_training = True
        self.go_to_visualisation_button.on_click(self.show_visualisation)

    def reg_log(self, run):
        if self.current_algo == 'reg_log_def':
            filename = 'reg_log.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                current_df = pd.read_csv(fullname ,sep=';')

            else:
                current_df = pd.DataFrame(columns = ['penalty', 'dual', 'tol', 'C', 'fit_intercept',
                                                         'intercept_scaling', 'class_weight', 'random_state','solver',
                                                         'max_iter', 'multi_class', 'warm_start', 'l1_ratio', 'n_jobs',
                                                         'Mean CV train neg-MSE', 'Mean CV test neg-MSE',
                                                          'Mean CV fit time'])

            trained_before_df = current_df.loc[(current_df['penalty']=='l2')&
                                                               (current_df['dual']==False)&
                                                               (current_df['tol']==0.0001)&
                                                               (current_df['C']==1.0)&
                                                               (current_df['fit_intercept']==True)&
                                                               (current_df['intercept_scaling']==1.0)&
                                                               (current_df['class_weight']=='None')&
                                                               (current_df['solver']=='lbfgs')&
                                                               (current_df['max_iter']==100)&
                                                               (current_df['multi_class']=='auto')&
                                                               (current_df['warm_start']==False)&
                                                               (current_df['l1_ratio']=='None')]
            if len(trained_before_df) > 0:
                result = widgets.Output()
                with result:
                    display(trained_before_df)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_current_df = widgets.HBox([result], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_current_df])
                self.test_acc = trained_before_df['Mean CV test neg-MSE'].values[0]
            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                clf = LogisticRegression()
                scores = cross_validate(clf, self.X_train, self.y_train, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
                train_acc = scores['train_score'].mean()
                self.test_acc = scores['test_score'].mean()
                fit_time = scores['fit_time'].mean()

                
                if len(current_df) > 0:
                    if self.test_acc > current_df['Mean CV train neg-MSE'].max():
                        result = f'Mean CV train neg-MSE: {round(train_acc,5)}. <br/>Mean CV test neg-MSE:  {round(self.test_acc,5)}. <br/>Best mean CV test neg-MSE acc for Logistic Regression so far!'
                    else:
                        result = f'Mean CV train neg-MSE: {round(train_acc,5)}. <br/>Mean CV test neg-MSE:  {round(self.test_acc,5)}.'
                else:
                    result = f'Mean CV train neg-MSE: {round(train_acc,5)}. <br/>Mean CV test neg-MSE:  {round(self.test_acc,5)}. <br/>First Logistic Regression model trained!'
                    
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is:  <br/>'+ result)])
                current_df = current_df.append({'penalty': 'l2', 'dual': False, 'tol': 0.0001,
                                                        'C': 1.0, 'fit_intercept': True,
                                                        'intercept_scaling': 1.0, 'class_weight': str(None), 'random_state': None,
                                                        'solver': 'lbfgs', 'max_iter': 100,'multi_class': 'auto',
                                                        'warm_start': False, 'l1_ratio': str(None), 'n_jobs': str(None),
                                                        'Mean CV train neg-MSE': round(train_acc, 5),
                                                        'Mean CV test neg-MSE': round(self.test_acc, 5),
                                                        'Mean CV fit time': round(fit_time, 5)},
                                                       ignore_index = True)
                current_df.to_csv(fullname, sep=';', index=False)

        elif self.current_algo == 'reg_log_sup':
            filename = 'reg_log.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                current_df = pd.read_csv(fullname ,sep=';')

            else:
                current_df = pd.DataFrame(columns = ['penalty', 'dual', 'tol', 'C', 'fit_intercept',
                                                         'intercept_scaling', 'class_weight', 'random_state', 'solver',
                                                         'max_iter', 'multi_class', 'warm_start', 'l1_ratio', 'n_jobs', 
                                                         'Mean CV train neg-MSE', 'Mean CV test neg-MSE',
                                                          'Mean CV fit time'])

            penalty=self.current_hp_box.children[0].value
            c=self.current_hp_box.children[1].value

            trained_before_df = current_df.loc[(current_df['penalty']==penalty)&
                                                               (current_df['C']==c)]
            if len(trained_before_df) > 0:
                result = widgets.Output()
                with result:
                    display(trained_before_df)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_current_df = widgets.HBox([result], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_current_df])
                self.test_acc = trained_before_df['Mean CV test neg-MSE'].values[0]
            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                try:
                    clf = LogisticRegression(penalty=penalty, C=c)
                    scores = cross_validate(clf, self.X_train, self.y_train, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
                    train_acc = scores['train_score'].mean()
                    self.test_acc = scores['test_score'].mean()
                    fit_time = scores['fit_time'].mean()
                    
                except ValueError as err:
                    self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                    [widgets.HTML(str(err))])
                    return

                if len(current_df) > 0:
                    if self.test_acc > current_df['Mean CV train neg-MSE'].max():
                        result = f'Mean CV train neg-MSE: {round(train_acc,5)}. <br/>Mean CV test neg-MSE:  {round(self.test_acc,5)}. <br/>Best mean CV test neg-MSE for Logistic Regression so far!'
                    else:
                        result = f'Mean CV train neg-MSE: {round(train_acc,5)}. <br/>Mean CV test neg-MSE:  {round(self.test_acc,5)}.'
                else:
                    result = f'Mean CV train neg-MSE: {round(train_acc,5)}. <br/>Mean CV test neg-MSE:  {round(self.test_acc,5)}. <br/>First Logistic Regression model trained!'
                    
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is:  <br/>'+ result)])
                current_df = current_df.append({'penalty': penalty, 'dual': False, 'tol': 0.0001,
                                                        'C': c, 'fit_intercept': True,
                                                        'intercept_scaling': 1.0, 'class_weight': str(None), 'random_state': None,
                                                        'solver': 'lbfgs', 'max_iter': 100,'multi_class': 'auto',
                                                        'warm_start': False, 'l1_ratio': str(None), 'n_jobs': str(None),
                                                        'Mean CV train neg-MSE': round(train_acc, 5),
                                                        'Mean CV test neg-MSE': round(self.test_acc, 5),
                                                        'Mean CV fit time': round(fit_time, 5)},
                                                       ignore_index = True)
                current_df.to_csv(fullname, sep=';', index=False)
        elif self.current_algo == 'reg_log_pro':
            filename = 'reg_log.csv'
            outdir = './trained_models/'+self.dataset_name
            fullname = os.path.join(outdir, filename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            if os.path.isfile(fullname):
                current_df = pd.read_csv(fullname ,sep=';')

            else:
                current_df = pd.DataFrame(columns = ['penalty', 'dual', 'tol', 'C', 'fit_intercept',
                                                         'intercept_scaling', 'class_weight', 'random_state','solver',
                                                         'max_iter', 'multi_class', 'warm_start', 'l1_ratio',
                                                         'Mean CV train neg-MSE', 'Mean CV test neg-MSE',
                                                          'Mean CV fit time'])

            penalty=self.current_hp_box.children[0].children[0].children[0].children[1].value
            dual=self.current_hp_box.children[0].children[1].children[0].children[1].value
            tol= self.current_hp_box.children[0].children[1].children[1].children[1].value
            c= self.current_hp_box.children[0].children[1].children[2].children[1].value
            fit_intercept= self.current_hp_box.children[0].children[0].children[1].children[1].value
            intercept_scaling=self.current_hp_box.children[0].children[1].children[3].children[1].value
            class_weight=self.current_hp_box.children[0].children[0].children[2].children[1].value
            try:
                random_state = int(self.current_hp_box.children[0].children[1].children[4].children[1].value)
            except ValueError:
                random_state = None
            solver=self.current_hp_box.children[0].children[0].children[3].children[1].value
            max_iter= self.current_hp_box.children[0].children[1].children[5].children[1].value
            multi_class=self.current_hp_box.children[0].children[0].children[4].children[1].value
            verbose=self.current_hp_box.children[0].children[1].children[6].children[1].value
            warm_start=self.current_hp_box.children[0].children[1].children[7].children[1].value
            try:
                n_jobs = int(self.current_hp_box.children[0].children[1].children[8].children[1].value)
            except ValueError:
                n_jobs = None
            try:
                l1_ratio = float(self.current_hp_box.children[0].children[1].children[9].children[1].value)
            except ValueError:
                l1_ratio = None

            trained_before_df = current_df.loc[(current_df['penalty']==penalty)&
                                                               (current_df['dual']==dual)&
                                                               (current_df['tol']==tol)&
                                                               (current_df['C']==c)&
                                                               (current_df['fit_intercept']==fit_intercept)&
                                                               (current_df['intercept_scaling']==intercept_scaling)&
                                                               (current_df['class_weight']==str(class_weight))&
                                                               (current_df['solver']==solver)&
                                                               (current_df['max_iter']==max_iter)&
                                                               (current_df['multi_class']==multi_class)&
                                                               (current_df['warm_start']==warm_start)&
                                                               (current_df['l1_ratio']==str(l1_ratio))]
            if len(trained_before_df) > 0:
                result = widgets.Output()
                with result:
                    display(trained_before_df)
                box_layout = Layout(display='flex', flex_flow='row', justify_content='space-around',width='auto')
                box_current_df = widgets.HBox([result], layout=box_layout)
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('This set of hyperparameters has already beed used for training.')]+[box_current_df])
                self.test_acc = trained_before_df['Mean CV test neg-MSE'].values[0]
            else:
                self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                [widgets.HTML('Creating classifier and starting training...')])
                try:
                    clf = LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=c,
                                                         fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                                                         class_weight=class_weight, random_state=random_state, solver=solver,
                                                         max_iter=max_iter, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs,
                                                         multi_class=multi_class, l1_ratio=l1_ratio)
                    scores = cross_validate(clf, self.X_train, self.y_train, cv=5, scoring='neg_mean_squared_error', return_train_score=True, error_score='raise')
                    train_acc = scores['train_score'].mean()
                    self.test_acc = scores['test_score'].mean()
                    fit_time = scores['fit_time'].mean()
                    
                except ValueError as err:
                    self.container.children = tuple(list(self.container.children)[:self.training_level] +
                                                    [widgets.HTML(str(err))])
                    return

                if len(current_df) > 0:
                    if self.test_acc > current_df['Mean CV train neg-MSE'].max():
                        result = f'Mean CV train neg-MSE: {round(train_acc,5)}. <br/>Mean CV test neg-MSE:  {round(self.test_acc,5)}. <br/>Best mean CV test neg-MSE for Logistic Regression so far!'
                    else:
                        result = f'Mean CV train neg-MSE: {round(train_acc,5)}. <br/>Mean CV test neg-MSE:  {round(self.test_acc,5)}.'
                else:
                    result = f'Mean CV train neg-MSE: {round(train_acc,5)}. <br/>Mean CV test neg-MSE:  {round(self.test_acc,5)}. <br/>First Logistic Regression model trained!'
                    
                self.container.children = tuple(list(self.container.children)[:self.training_level+1] +
                                                [widgets.HTML('Done! The result is:  <br/>'+ result)])
                current_df = current_df.append({'penalty': penalty, 'dual': dual, 'tol': tol,
                                                        'C': c, 'fit_intercept': fit_intercept,
                                                        'intercept_scaling': intercept_scaling, 'class_weight': str(class_weight),
                                                        'random_state':random_state, 'solver': solver, 'max_iter': max_iter,'multi_class': multi_class,
                                                        'warm_start': warm_start, 'l1_ratio': str(l1_ratio), 'n_jobs': str(n_jobs),
                                                        'Mean CV train neg-MSE': round(train_acc, 5),
                                                        'Mean CV test neg-MSE': round(self.test_acc, 5),
                                                        'Mean CV fit time': round(fit_time, 5)},
                                                       ignore_index = True)
                current_df.to_csv(fullname, sep=';', index=False)
        
        fig_widget = scatter_plot('Logistic Regression', self.test_acc, self.dataset_name, False)
        self.container.children = tuple(list(self.container.children)[:self.plotting_level] +
                                            fig_widget) 
        self.container.children = tuple(list(self.container.children)[:self.plotting_level+1] +
                                            [self.go_to_visualisation_button]) 
        self.from_training = True
        self.go_to_visualisation_button.on_click(self.show_visualisation)
