import streamlit as st

# def main():
#     st.title('prima app')

# if __name__ == "__main__":
#     main()
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder as le
import matplotlib.pyplot as plt
import io
from summarytools import dfSummary




#@st.cache_data
def load_data():
    st.title("Data Transformation")
    uploaded_file = st.file_uploader("Choose a file")#,type=["xlsx", "csv"])
    df=pd.read_csv(uploaded_file)
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
    # target=df['class']
    # df['class']=le().fit_transform(target)
    # model_pipe=joblib.load('iris_pipe.pkl')
    model_pipe=joblib.load(st.file_uploader("insert the pipe model"))

    return df,model_pipe

def convert_to_excel(df):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine="xlsxwriter")
    df.to_excel(writer, sheet_name="data",index=False)
    # see: https://xlsxwriter.readthedocs.io/working_with_pandas.html
    writer.close()
    return output.getvalue()

# def create_pairplot(data, features, hue=None):

#     #fig = plt.figure(figsize=(12,12))
#     fig = sns.countplot(data[features], hue=hue, diag_kind='kde',height=1)

#     """Crea un pairplot con le features selezionate"""
#     #fig = plt.figure(figsize=(12,12))
#     fig = sns.pairplot(data[features], hue=hue, diag_kind='kde',height=1)

#     """Crea un pairplot con le features selezionate"""
#     #fig = plt.figure(figsize=(12,12))
#     fig = sns.relplot(data[features], hue=hue, diag_kind='kde',height=1)
    
#     return fig

# def heatmap(data):
#     fig1,axes=plt.subplot()
#     sns.heatmap(data.corr(numeric_only=True), cmap='coolwarm',ax=axes)
#     return fig1

def main():
    st.title('Analisi del Dataset IRIS')

    df,model_pipe = load_data()
    st.dataframe(df)

    tab1,tab2,tab3=st.tabs(['Analisi','Prediction','Drag & Drop'])

    with tab1:

        st.subheader(' Descrizione dei Dati')
        st.dataframe(df.describe().T)
        st.subheader(' Summary dei Dati')
        st.dataframe(dfSummary(df))

        st.subheader(' correlation dei Dati')
        st.dataframe(df.corr(numeric_only=True))
        # fig,axes=plt.subplot()
        # fig1=heatmap(df)
        # st.pyplot(heatmap(df))

    # # with st.sidebar:
    #     features= st.multiselect(
    #     'Seleziona le variabili da visualizzare',
    #     options=iris.columns.tolist(),
    #     default=['sepal length', 'sepal width', 'petal length', 'petal width']#df1.drop(['class'], axis=1)
    #     )

    #     hue_var = st.selectbox(
    #                             'Seleziona la variabile per il colore',
    #                             options=[None] + [col for col in iris.columns if iris[col].nunique() <= 5],
    #                             help='Questa variabile verrà usata per colorare i punti nel pairplot'
    #                             )        

    
    # st.subheader(' Countplot delle Variabili')
    # with st.spinner('Generazione del pairplot in corso...'):
        # plt.figure(figsize=(10,7))
        # axes=plt.subplot(df)
        # fig = sns.countplot(df, x='sepal width', hue='class',ax=axes)
        # st.pyplot(fig)

        st.subheader(' Pairplot delle Variabili')
        with st.spinner('Generazione del pairplot in corso...'):
            fig = sns.pairplot(df,  hue='class',)
            st.pyplot(fig)
        
        st.subheader(' Implot delle Variabili')
        with st.spinner('Generazione del implot in corso...'):
            fig = sns.lmplot(x="petal length", y="petal width",hue='class', data=df, fit_reg=True)
            st.pyplot(fig)

    with tab2:
        st.subheader('Data prediction')
        st.spinner('generazione della prediction in corso ...')

        sepal_length=st.slider('sepal length',0.0,10.0,2.5)
        sepal_width=st.slider('sepal width',0.0,10.0,2.5)
        petal_length=st.slider('petal length',0.0,10.0,2.5)
        petal_width=st.slider('petal width',0.0,10.0,2.5)

        data = {
            "sepal length":[sepal_length],
            "sepal width": [sepal_width],
            "petal length": [petal_length],
            "petal width": [petal_width],
                }
        
        input_df=pd.DataFrame(data)
        
        if st.button('Prediction'):
            res = model_pipe.predict(input_df).astype(int)[0]
            classes = {0:'setosa',
            1:'versicolor',
            2:'virginica'
                }
            # print(res)
            y_pred = classes[res]
            st.success(y_pred)
    
    with tab3:
        uploaded_file = st.file_uploader("Choose a file",type=["xlsx", "csv"])
        if uploaded_file is not None:
         # Verifica l'estensione del file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                st.success("File CSV caricato con successo!")
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
                st.success("File XLSX caricato con successo!")
            else:
                st.error("Formato file non supportato!")
        
        if st.button('Start Processing', help="Process Dataframe"):
            res = model_pipe.predict(input_df).astype(int)[0]
            classes = {0:'setosa',
            1:'versicolor',
            2:'virginica'
                }
            # print(res)
            y_pred = classes[res]
            st.header('Addes Column')
            df['predicted'] = y_pred
            st.dataframe(df)
            st.balloons()
            st.download_button(
                                label="download as Excel-file",
                                data=convert_to_excel(df),
                                file_name=f"data_predited.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="excel_download",
                                )


if __name__ == "__main__":
    main()