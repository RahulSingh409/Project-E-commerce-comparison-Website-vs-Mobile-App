#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
from func import FeatureSelector
from charts import *
from models import ModelRunner
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import random
import pickle
import streamlit as st
import seaborn as sns
from sklearn.linear_model import Ridge
from PIL import Image
import joblib
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport
import sweetviz as sv
import streamlit.components.v1 as components
import matplotlib
import matplotlib.pyplot as plt


#load model
pickle_in = open("ecommerce.pkl","rb")
ecommerce= joblib.load(pickle_in)

def welcome():
    return "Welcome All"

def user_input_features(Avg_Session_length, Time_on_App, Time_on_Website, Length_of_MemberShip):  
    prediction= ecommerce.predict([[Avg_Session_length, Time_on_App, Time_on_Website, Length_of_MemberShip]])
    print(prediction)
    return prediction


def main():
    # import data file csv
    df = pd.read_csv('Ecommerce.csv')
    # set page title
    st.set_page_config('Fashion Ecommerce Industry')

    st.title('Customer Annual Spent Prediction')
    #import the image
    # Using "with" notation
    with st.sidebar:
        image= Image.open("iidt_logo_137.png")
        add_image=st.image(image,use_column_width=True)

    social_acc = ['About']
    social_acc_nav = st.sidebar.selectbox('About', social_acc)
    if social_acc_nav == 'About':
        st.sidebar.markdown("<h2 style='text-align: center;'> This Project completed under ExcelR, the team completed the project:</h2> ", unsafe_allow_html=True)
        st.sidebar.markdown('''---''')
        st.sidebar.markdown('''
        • Mr. Rahul Kumar Singh \n 
        • Mr. R vamsi Sai Kumar reddy \n 
        • Mr. Vishal Vijay Kakade \n
        • Mr. R. Hari Haran \n 
        • Mr. Mummareddy Manikumara Swamy \n
        • Mr. Sushilkumar Yadav''')
        
    menu_list = ['Feature Data Analysis','Exploratory Data Analysis' , "Predict Annual Spent per Customer"]
    menu = st.radio("Menu", menu_list)

    if menu == 'Feature Data Analysis':
        st.title('Feature Data Analysis of Ecommerce company ')

        if st.checkbox("View data"):
            st.write(df)
        #select target variable
        target = st.selectbox("Select Target Feature",df.columns)

        #select feature selection method
        selector = st.radio(label="Selection Method",options=["RFE","SelectFromModel"])
        F = FeatureSelector(df,target)
        univariate,ref,sfm,problem = F.get_result_dictionaries()
            #chart
        if selector == "RFE":
              fig = barchart(ref["feature_names"], ref["ranking"],"Ranking acc to RFE")
        elif selector == "SelectFromModel":
              fig = barchart(sfm["feature_names"], sfm["scores"],"Feature Scores acc to SelectFromModel")
        st.pyplot(fig)

         #select k number of features to proceed
        k = st.number_input("Number of Feature to proceed (k): ", min_value=0, max_value= len(df.columns) - 1)
        if problem == "regression":
            model = st.selectbox("ML Method",["Linear Regression","XGBoost","ElasticNetCV","Decision Tree","SGDRegressor","Ridge","Lasso","RandomForestRegressor","GradientBoostingRegressor","Support Vector Machine"])
        else:
             model = st.selectbox("ML Method",["Logistic Regression","Decision Tree","SGDRegressor","RandomForestRegressor","GradientBoostingRegressor"])
        #when k is determined 
        if k > 0:
            #get last X,y according to feature selection
            X,_,temp,col_types,_ = F.extract_x_y() 
            y = df[target].values.reshape(-1,1)
            #feature set
            if selector == "SelectKBest":
                X = F.univariate_feature_selection(X,y,temp,k)["X"]
            elif selector == "RFE":
                X = F.ref_feature_selection(X,y,temp,col_types,k)["X"]
            elif selector == "SelectFromModel":
                X = F.sfm_feature_selection(X,y,temp,col_types,k)["X"]
            #run models
            M = ModelRunner(model,X,y,problem)
            score = M.runner()
            #display score
            st.write("Score of Model: {}".format(score))
    
    elif menu == 'Exploratory Data Analysis':
        EDA_menu_list = ['None','Pandas Profiling', 'Sweetviz', 'EDA Explanation']
        menu = st.selectbox("EDA Menu", EDA_menu_list)
        
        if menu == 'None':
            return None
        
        if menu == 'Pandas Profiling':
            st.title('Automating EDA using Pandas Profiling of Ecommerce company ')
            
            profile = ProfileReport(df,title="Agriculture Data",

                            dataset={

                            "description": "This profiling report is done for Ecommerce Industry, its Feature Importance.",
                          },
                variables={

                    "descriptions": {

                        "Customer ID": "A unique identification number given to every Customer.",

                        "Avg Session length": "Customers come in to the store, have sessions/meetings with a personal stylist.",

                        "Time on App": "Time spent by customer on company mobile app.",

                        "Time on Website": "Time spent by customer on website.",

                        "Length of MemberShip": "The number of years/months in a membership term by customer.",

                        "Yealy amount spent": "Amount yearly spent by customer in an ecommerce shopping with the company.",
                      
                            }

                        }

                    )
            st.write(df)

            st_profile_report(profile)
            
        if menu == 'Sweetviz':
            st.title('Automating EDA using Sweetviz of Ecommerce company ')
            if st.checkbox("View data"):
                st.write(df)
            
            sweet_report = sv.analyze(df)
            sweet_report.show_html('sweet_report.html')
            components.iframe(src='http://localhost:3001/sweet_report.html', width=1100, height=1200, scrolling=True)
            
        
        if menu == 'EDA Explanation':
            st.title('Exploratory Data Analysis of Ecommerce Industry')
            if st.checkbox("View data"):
                st.write(df)
            
            st.markdown('---')
            st.markdown("<h2 style='text-align: center;'> Visualisation and Analysis </h2>", unsafe_allow_html=True)
            st.markdown('---')
            st.markdown("<h4 style='text-align: left;'> A) VARIABLE DESCRIPTIONS:</h4>", unsafe_allow_html=True)
            st.markdown('''
            We know we're working with 623 observations of 6 variables:\n
            1. Customer ID: A unique identification number given to every Customer.\n
            2. Avg Session length: Customers come in to the store, have sessions/meetings with a personal stylist.\n
            3. Time on App: Time spent by customer on company mobile app.\n
            4. Time on Website: Time spent by customer on website.\n
            5. Length of MemberShip:The number of years/months in a membership term by customer.\n
            6. Yealy amount spent:Amount yearly spent by customer in an ecommerce shopping with the company.\n 
            **As per our Company Business Problem or Business objective:****The company is trying to decide whether to focus their efforts 
            on their mobile app experience or their website. The company annual income will depend on how much customer spent in 
            company.**\n
            **So, 'Yearly amount spent' by customer will be our target variable.** 
            ''')

            st.markdown("<h4 style='text-align: left;'> B) Some key information about the variables </h4>", unsafe_allow_html=True)
            st.image('describe.png')
            st.markdown("<h4 style='text-align: left;'> Insights </h4>", unsafe_allow_html=True)
            st.markdown('''
            The minimum timing that a customer spend at the store is 30 min and the maximum is 35 min. The minimum average time spent on  
            mobile app is 9 min and the maximum is 15 min, time spent on website mean is 37 min that is much more than that of App. The std             shows that data are less despersed in relation to the mean. The max increase in yearly amount spent shows that the data is not  
            normally distributed, there is skewness in the dataset.
            ''')

            st.markdown("<h4 style='text-align: left;'> C)  Correlation Heatmap </h4>", unsafe_allow_html=True)
            st.image('download.png')
            st.markdown("<h4 style='text-align: left;'> Inference from ‘r’ values and heat map </h4>", unsafe_allow_html=True)
            st.markdown('''
            * No 2 factors have strong linear relationships. \n
            * Avg Session length, Time on App, Length of MemberShip are all negatively correlated with Yearly amount spent. Have moderate 
            negative relationship.\n
            * Only Time on Website is positively related with all others variables.\n
            * 'Customer ID' is uniformaliy distributed and there is no correlation with other variable.
            ''')
            
            st.markdown("<h4 style='text-align: left;'>D)  Feature Exploration on Distplot and Box Plot  </h4>", unsafe_allow_html=True)
            st.image('download (1).png')
            st.image('download (2).png')
            st.image('download (3).png')
            st.image('download (4).png')
            st.image('download (5).png')
            st.markdown("<h4 style='text-align: left;'> Inference from Distplot and Box Plot </h4>", unsafe_allow_html=True)
            st.markdown('''
            (Left-distplot) all the variables are normally distributed, and there are little skewness in the dataset. \n
             For the (Right-Boxplot)   we can see 'Yealy amount spent ' & 'Length of MemberShip' consists lot of outliers present below the 
             minimum and the maximum.
            ''')

            st.markdown("<h4 style='text-align: left;'> E) Plot different feature on scatter plot </h4>", unsafe_allow_html=True)
            st.image('download (6).png')
            st.markdown("<h4 style='text-align: left;'> Inference </h4>", unsafe_allow_html=True)
            st.markdown('''
            From scatter plots, we can see there is no positive nor negative linear relationships between the variable.
            ''')

            st.markdown("<h4 style='text-align: left;'> F) Feature Exploration on Pairplot </h4>", unsafe_allow_html=True)
            st.image('download (7).png')
            st.markdown("<h4 style='text-align: left;'> Inference from Pair Plots </h4>", unsafe_allow_html=True)
            st.markdown(''' 
            * From scatter plots, we can see there is no positive nor negative linear relationships between the variable.\n
            * For distribution all are normally distributed.
            ''')

            st.markdown("<h4 style='text-align: left;'> G) Feature Exploration on Regplot </h4>", unsafe_allow_html=True)
            st.image('download (8).png')
            st.markdown("<h4 style='text-align: left;'> Inference from Regplot </h4>", unsafe_allow_html=True)
            st.markdown('''
            From above regplot, we have compare each variables with the target variable. Only 'Time on Website' is giving positive response 
            as it increases.'Length of Membership' is uniform to our target variable. Rest all other variables are showing negative relation 
            with target variable.
            ''')

            st.markdown("<h4 style='text-align: left;'> H) Plot different features against one another Barplot using bin. </h4>", unsafe_allow_html=True)
            st.image('download (9).png')
            st.image('download (10).png')
            st.image('download (11).png')
            st.image('download (14).png')
            st.markdown("<h4 style='text-align: left;'> Inference from Bin Size </h4>", unsafe_allow_html=True)
            st.markdown('''
            As the time on app is increasing the amount spent by customer is decreasing. Time on website is increasing the customer is also 
            spending more. This shows that company website is generating more revenue than the app.
            ''')
            
            st.markdown("<h4 style='text-align: left;'> I) Feature importance </h4>", unsafe_allow_html=True)
            st.image('download (15).png')
            st.markdown('''
            We can clearly find out that Time on Website is the most important feature.
            ''')
            
            st.markdown("<h2 style='text-align: left;'> Conclusion & Recommendation </h2>", unsafe_allow_html=True)
            st.markdown('''
            * Most of the variables are negatively correlated with each other.\n
            * Only Time on Website is positively related with all others variables.\n
            * Both models and our data analysis shows that 'Website' generates more sales than 'Mobile App'. So company should focus there budget on website to make it more suitable for the customer.\n
            * To increase the company sale from mobile app company should try to make discount chart, if customer order more product from company app than more discount be allowed.\n
            * While doing EDA, we have seen 'Length of Membership' and ‘Yearly amount spent' is showing uniformity. Company has to see, how to increase customer spending although they are loyal customer but company has to find a way like giving them extra discount or pre sale entry to increase there spending with the company.\n
            * Lastly, **So as Time on Website is a much more significant factor than Time on App, the company has a choice: they could either focus all the attention into the Website as that is what is bringing the most money in, or they could focus on the App as it is performing so poorly! I would recommend company to spent more on website and make it more smooth but also increase the user exp. on mobile app as well.**
            ''')
                
    elif menu == 'Predict Annual Spent per Customer':
            
            st.title("Fashion Ecommerce Industry")
            #import the image
            image= Image.open("Industry-11.png")
            st.image(image,use_column_width=True)

            html_temp = """
            <div style="background-color:tomato;padding:10px">
            <h2 style="color:white;text-align:center;">Customer Annual Spent Prediction </h2>
            </div>
            """
            st.markdown(html_temp,unsafe_allow_html=True)
            Avg_Session_length = st.text_input("Avg_Session_length","Type Here")
            Time_on_App = st.text_input("Time_on_App","Type Here")
            Time_on_Website = st.text_input("Time_on_Website","Type Here")
            Length_of_MemberShip = st.text_input("Length_of_MemberShip","Type Here")
            result=""
            if st.button("Predict"):
                result=user_input_features(Avg_Session_length, Time_on_App, Time_on_Website, Length_of_MemberShip)
            st.success('The predicted Annual spent is {}'.format(result))
            if st.button("About"):
                st.text("The amount Prediction is customer spent which leads to company annual income.")
                st.text("ExcelR Project")
            
if __name__=='__main__':
    main()
    
    
    

