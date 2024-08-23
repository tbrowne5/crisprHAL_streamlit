# TO BE DONE:
# • Add an option in the sidebar to remove duplicate sgRNAs or sgRNAs with 18/20 (PAM distal) matches
# • Add an option to check for sgRNA target sites in circular segments of DNA
# • Add an option to display the lowest predicted activity sgRNAs instead
# • Add a warning for trying to predict X number of sgRNA activites

import streamlit as st
from Bio import SeqIO
import io
import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

def load_model(modelname):
    model = tf.keras.models.load_model(modelname) #'Citro_TevSpCas9.h5')
    return model

def convert_base(arrayinput,padded=False):
    arrayoutput = []
    if padded:
        for sequence in arrayinput:
            sequence = "NNNNNNNNNN" + sequence + "NNNNN"
            onehotencoding = []
            for i in range(len(sequence)):
                if sequence[i].upper() == "T":
                    onehotencoding.append([1,0,0,0,0])
                elif sequence[i].upper() == "G":
                    onehotencoding.append([0,1,0,0,0])
                elif sequence[i].upper() == "C":
                    onehotencoding.append([0,0,1,0,0])
                elif sequence[i].upper() == "A":
                    onehotencoding.append([0,0,0,1,0])
                elif sequence[i].upper() == "N":
                    onehotencoding.append([0,0,0,0,0])
            arrayoutput.append(np.array(onehotencoding))
    else:
        for sequence in arrayinput:
            onehotencoding = []
            for i in range(len(sequence)):
                if sequence[i].upper() == "A":
                    onehotencoding.append([1,0,0,0])
                elif sequence[i].upper() == "C":
                    onehotencoding.append([0,1,0,0])
                elif sequence[i].upper() == "G":
                    onehotencoding.append([0,0,1,0])
                elif sequence[i].upper() == "T":
                    onehotencoding.append([0,0,0,1])
                elif sequence[i].upper() == "N":
                    onehotencoding.append([0,0,0,0])
            arrayoutput.append(np.array(onehotencoding))
    return np.array(arrayoutput)

def find_sgRNA_sequences(fasta_content):
    # Assuming the sgRNA targets are to be 20 nucleotides long
    target_length = 28
    targets = []
    
    for record in SeqIO.parse(fasta_content, "fasta"):
        sequence = str(record.seq)
        # Sliding window to find all possible sgRNA targets of specified length
        for i in range(len(sequence) - target_length + 1):
            sgRNA = sequence[i:i + target_length]
            # Example condition: sgRNA must start with 'G'
            if sgRNA[21:23] == "GG":
                targets.append(sgRNA)
        sequence = str(record.seq.reverse_complement())
        # Sliding window to find all possible sgRNA targets of specified length
        for i in range(len(sequence) - target_length + 1):
            sgRNA = sequence[i:i + target_length]
            # Example condition: sgRNA must start with 'G'
            if sgRNA[21:23] == "GG":
                targets.append(sgRNA)

    return targets

# CURRENTLY TESTING THIS FUNCTION TO DOWNLOAD FILES...
def output_processing(np_train, y_pred):
    np_train = np.array(np_train)
    outputreturn = {}
    for i in range(0,len(np_train)): outputreturn[np_train[i][0:20]] = str(format(float(y_pred[i]), '.8f')).replace(' [','').replace('[', '').replace(']', '')
    return dict(sorted(outputreturn.items(), key=lambda item: float(item[1]), reverse=True))

def output_download(df):
    
    # Sort DataFrame in descending order by the first column as an example
    #sorted_df = df.sort_values(by=0, ascending=False)
    # Convert DataFrame to CSV
    csv = df.to_csv(index=True)
    # Create a download link
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name='sorted_sgRNA_targets.csv',
        mime='text/csv',
    )

def generate_plot(preds, modelname):
    data = np.array(preds)
    #st.write(np.max(data))

    # Create a density plot using Seaborn
    fig2, ax2 = plt.subplots()
    
    backgroundalpha = 1.0
    ax2.axvspan(-3.5, -2.5, color='#D9DD17', alpha=backgroundalpha)    # Coloring the background from -3 to -2
    ax2.axvspan(-2.5, -1.5, color='#A0D52F', alpha=backgroundalpha) # Coloring the background from -2 to -1
    ax2.axvspan(-1.5, -0.5, color='#70C94B', alpha=backgroundalpha)  # Coloring the background from -1 to 0
    ax2.axvspan(-0.5, 0.5, color='#4ABB5F', alpha=backgroundalpha)    # Coloring the background from 0 to 1
    ax2.axvspan(0.5, 1.5, color='#2DA86F', alpha=backgroundalpha)    # Coloring the background from 0 to 1
    ax2.axvspan(1.5, 2.5, color='#1D9677', alpha=backgroundalpha)     # Coloring the background from 1 to 2
    ax2.axvspan(2.5, 3.5, color='#1B847D', alpha=backgroundalpha)   # Coloring the background from 2 to 3
    
    #sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
    sns.histplot(data, kde=True, color='black', ax=ax2)
    labels = ["Rare","Very Low","Low","Moderate","High","Very High","Rare"]
    for i in range(0,7):
        ax2.annotate(labels[i], xy=(((i)/7)+(1/14), 0.915), xycoords='axes fraction', xytext=(0, 10), 
             textcoords='offset points', ha='center', va='bottom',
             fontsize=8, color='black')
    plt.axvline(x=float(np.max(data)), color='red', linestyle='--', linewidth=2)
    plt.annotate('Best', xy=(float(np.max(data))+0.4, 0.9), xycoords='data', xytext=(0, 10), 
             textcoords='offset points', ha='center', va='bottom',
             fontsize=12, color='red')
    ax2.legend([],[], frameon=False)
    ax2.set_xlim(-3.5, 3.5)
    ax2.set_title('How to interpret the ' + str(modelname) + ' model predictions')
    ax2.set_xlabel('Predicted activity scores')
    ax2.set_ylabel('Density')
    
    #params = {"ytick.color" : "w",
    #      "xtick.color" : "w",
    #      "axes.labelcolor" : "w",
    #      "axes.edgecolor" : "w"}
    #plt.rcParams.update(params)
    
    ax2.xaxis.label.set_color('white')
    ax2.yaxis.label.set_color('white')
    ax2.title.set_color('white')

    # Change the color of all spine elements on the graph
    for spine in ax2.spines.values():
        spine.set_color('white')

    # Set tick colors
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')

    # Change the legend text to white
    legend = ax2.get_legend()
    for text in legend.get_texts():
        text.set_color('white')
    
    plt.savefig("seaborn_plot.png", bbox_inches='tight', transparent=True)
    plt.close()

    # Display the plot
    col1.image("seaborn_plot.png", use_column_width=True)
    #st.pyplot(fig2)

st.title('crisprHAL')
st.subheader('A generalizable bacterial Cas9/sgRNA prediction model')
uploaded_file = st.file_uploader("Upload a FASTA file", type=["fasta","txt"])

option = st.sidebar.selectbox(
    'Select the nuclease:',
    ('TevSpCas9', 'SpCas9','Citrobacter TevSpCas9')
)

crisprHALtevspcas9info = """
Model:
<br>• Name: crisprHAL, TevSpCas9
<br>• Paper: https://www.nature.com/articles/s41467-023-41143-7
<br>• Source Code: [Github/tbrowne5/crisprHAL](https://github.com/tbrowne5/crisprHAL)
"""

crisprHALsource = """
Source Code: [Github/tbrowne5/crisprHAL](https://github.com/tbrowne5/crisprHAL)
"""

guoespcas9datasource = """
Base Model Training Data: <br>• <i>[Escherichia coli, eSpCas9](https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA450978)</i>
"""

crisprHALtevspcas9datasource = """
Transfer Learning Training Data:<br>• <i>[Escherichia coli, TevSpCas9](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA939699)</i>
"""

crisprHALspcas9datasource = """
Transfer Learning Training Data:<br>• <i>[Escherichia coli, SpCas9](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA939699)</i>
"""

crisprHAL2tevspcas9datasource = """
Training Data:<br>• <i>Citrobacter rodentium, TevSpCas9</i>
"""

if option == 'TevSpCas9':
    st.sidebar.title('TevSpCas9 Model')
    st.sidebar.write('Enzyme: TevSpCas9')
    st.sidebar.write('Model: crisprHAL')
    st.sidebar.markdown(crisprHALsource, unsafe_allow_html=True)
    st.sidebar.markdown(guoespcas9datasource, unsafe_allow_html=True)
    st.sidebar.markdown(crisprHALtevspcas9datasource, unsafe_allow_html=True)
elif option == 'SpCas9':
    st.sidebar.title('SpCas9 Model')
    st.sidebar.write('Enzyme: SpCas9')
    st.sidebar.write('Model: crisprHAL')
    st.sidebar.markdown(crisprHALsource, unsafe_allow_html=True)
    st.sidebar.markdown(guoespcas9datasource, unsafe_allow_html=True)
    st.sidebar.markdown(crisprHALspcas9datasource, unsafe_allow_html=True)
elif option == 'Citrobacter TevSpCas9':
    st.sidebar.title('Citro. TevSpCas9 Model')
    st.sidebar.write('Enzyme: TevSpCas9')
    st.sidebar.write('Model: crisprHAL 2.0')
    st.sidebar.markdown(crisprHAL2tevspcas9datasource, unsafe_allow_html=True)
    st.sidebar.write('Note: crisprHAL 2.0 is still in development')
footer = """
<style>
.footer {
    color: white;                 /* White text color */
    position: fixed;              /* Fixed position to stay at the bottom */
    left: 2px;                      /* Align to the left edge */
    bottom: 0px;                    /* Align to the bottom edge */
    height: 80px;                 /* Footer height */
    /*width: calc(100% - 260px);*/    /* Footer width accommodating the sidebar */
    width: 100%;                 /* Footer width */
    background-color: #00000D;      /* Colour for background */
    text-align: left;          /* Align text inside the footer to the right */
    padding-left: 350px;        /* Left padding for the text */
    padding-right: 75px;        /* Right padding for the text */
    font-size: 1em;
    line-height: 25px;            /* Center text vertically */
    overflow: hidden;
}

@media (max-width: 765px) {
    .footer {
        padding-left: 25px;        /* Reduce left padding for smaller screens or when sidebar is likely hidden */
    }
}

</style>
<div class="footer">
    <div class="footer-content">
        <b>CITATION:</b><br>Ham, D.T., Browne, T.S., Banglorewala, P.N. et al. A generalizable Cas9/sgRNA prediction model using machine transfer learning with small high-quality datasets. <em>Nat Commun</em> <b>14</b>, 5514 (2023). <a>doi.org/10.1038/s41467-023-41143-7</a>
    </div>
</div>
"""
# CITATION:
#Ham, D.T., Browne, T.S., Banglorewala, P.N. et al. A generalizable Cas9/sgRNA prediction model using machine transfer learning with small high-quality datasets. <em>Nat Commun</em> <b>14</b>, 5514 (2023). [https://doi.org/10.1038/s41467-023-41143-7](https://doi.org/10.1038/s41467-023-41143-7)</p>

st.markdown(footer, unsafe_allow_html=True)

if uploaded_file is not None:
    fasta_content = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
    targets = find_sgRNA_sequences(fasta_content)
    if targets:
        if option == 'TevSpCas9':
            modelinputs = convert_base(targets, True)
            model = load_model('TevSpCas9.h5')
        elif option == 'SpCas9':
            modelinputs = convert_base(targets, True)
            model = load_model('SpCas9.h5')
        elif option == 'Citrobacter TevSpCas9':
            modelinputs = convert_base(targets, False)
            model = load_model('Citro_TevSpCas9.h5')

        predictions = model.predict(modelinputs)
        #predictionsdf = pd.DataFrame(predictions, index=predictions[:,0])
        output = output_processing(targets,predictions)
        outputdf = pd.DataFrame.from_dict(output,orient='index')
        #st.write(outputdf)
        
        output_download(outputdf)

        col1, col2 = st.columns([3,2])

        generate_plot(predictions, option)
        
        col2.subheader("Best guides found")
        top=1
        for item in output:
            score = str(output[item])
            if len(score) > 6: score = score[0:5]
            col2.write("#" + str(top) + ": " + str(item[0:20]) + " " + score) #str(output[item]))
            top+=1
            if top > 5: break
        col2.write("Found " + str(len(output)) + " sgRNA target sites.")
        #st.write("\n\n\n\n\nALL OUTPUTS:")
        #for item in output:
        #    st.write(" sgRNA: " + str(item[0:20]) + ", score: " + str(output[item]))
    else:
        st.write("No valid sgRNA targets found.")
