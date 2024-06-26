import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import plotly.express as px
import plotly
import pickle
import time
from math import ceil
import re

from top2vec import Top2Vec

st.set_page_config(layout = "wide", page_title="4th_demo")

part_rem_serice = ["Bodyshop","Free Service","Pre-Sale/PDI","-"]
labour_rem_service = ["Bodyshop","Free Service","Pre-Sale/PDI","PDI SERVICE","1ST FREE SERVICE",
               "2ND FREE SERVICE","3RD FREE SERVICE","2NDTFREEMAINTSERVICE","1STFREEMAINTSERVICE","3RDFREEMAINTSERVICE",
               "4TH FREE SERVICE","5TH FREE SERVICE","6TH FREE SERVICE","FC WORK","PRESALE SHORTAGE", "PRESALE"]

sevice_category_map = {'Paid Service': 'Paid Service',
 'Free Service': 'Free Service',
 'Repair': 'Repair',
 'Bodyshop': 'Bodyshop',
 'Pre-Sale/PDI': 'Pre-Sale/PDI',
 'Accessories': 'Accessories',
 'En-Route': 'En-Route',
 'Free Maint Scheme': 'Free Maint Scheme',
# '-': '-',
 'RUNNING REPAIR': 'Repair',
 '1ST FREE SERVICE': 'Free Service',
 'ACCIDENTAL': 'Repair',
 'SERVICE ACTION': 'Paid Service',
 '2ND FREE SERVICE': 'Free Service',
 'PANEL REPAIR': 'Repair',
 '3RD FREE SERVICE': 'Free Service',
 'PAID SERVICE': 'Paid Service',
 'PRESALE': 'Pre-Sale/PDI',
 'PDI SERVICE': 'Pre-Sale/PDI',
 'Service Accessories': 'Accessories',
# 'FC WORK': 'FC WORK',
 'Sales Accessories': 'Accessories',
 'En-route Repair': 'En-Route',
 '4TH FREE SERVICE': 'Free Service',
 '2NDTFREEMAINTSERVICE': 'Free Service',
 '3RDFREEMAINTSERVICE': 'Free Service',
 'CAMPAIGN': 'CAMPAIGN',
 '1STFREEMAINTSERVICE': 'Free Service',
 '6TH FREE SERVICE': 'Free Service',
 'PRESALE SHORTAGE': 'Pre-Sale/PDI'}

#default_service = list(sevice_category_map.values())
default_service = ['Accessories', 'Paid Service', 'Repair', 'En-Route', 'Free Service']

diff_prob = {"brake":[],
"clutch":[],
"suspension":[],
"steering":[],
"engine":[],
"ac":[],
"infotainment":[],
"light":[],
"exhaust":[]}

for k,v in diff_prob.items():
    diff_prob[k] = set(diff_prob.keys())-set([k])

@st.cache_data
def load_required_variable():
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    df = pd.read_csv("output/verbatims_1_yr_data_after_merging_similar_action_taken_new.csv")
    df.action_taken = df.action_taken.apply(lambda x:eval(x))
    
    rem=[]
    for i in df.verbatim:
        if i.isdigit() or len(re.sub("[^A-Za-z0-9]", "", i))<=2:
            rem.append(i)
    
    df = df[~df.verbatim.isin(list(rem))]
    
    top2vec_model = Top2Vec.load("models/mahindra_2_0_top2vec_model_1_yr_data_after_filtering_and_unique_docs")
    #top2vec_model = Top2Vec.load("models/mahindra_2_0_top2vec_model_1_yr_data_after_filtering_and_unique_docs_all_minilm_l6_v2_model")
    
    with open("output/verbatims_1_yr_data_topics_combined_similarity_score_0_7_replaced_topic_map.pkl","rb") as f:
        replaced_topic_map = pickle.load(f)
    
    part_df = pd.read_csv("output/part_df_cleaned_1_yr_data_with_all_service_category_consumables_filtered.csv")
    
    #labour_df = pd.read_csv("output/labour_df_cleaned_1_yr_data_with_all_service_category_age_added.csv")
    labour_df = pd.read_csv("output/labour_df_cleaned_1_yr_data_with_all_service_category_age_added_consumables_filtered.csv")
    
    part_desc2id = dict(zip(part_df.PART_DESC,part_df.PART_NUMBR))
    labour_desc2id = dict(zip(labour_df.LABR_DESC,labour_df.LABR_CD))
    
    service_category_groups = {}
    for k, v in sevice_category_map.items():
        if v not in service_category_groups:
            service_category_groups[v]=[]
        service_category_groups[v].append(k)

    with open("output/verbatim_emb_cluster_wise_1_yr_data.pkl","rb") as f:
        verbatims_emb = pickle.load(f)

    with open("output/actions_emb_cluster_wise_1_yr_data.pkl","rb") as f:
        actions_emb = pickle.load(f)

    return model, df, top2vec_model, replaced_topic_map, part_df, labour_df, part_desc2id, labour_desc2id, service_category_groups, verbatims_emb, actions_emb

model,df,top2vec_model,replaced_topic_map,part_df,labour_df,part_desc2id,labour_desc2id,service_category_groups,verbatims_emb,actions_emb = load_required_variable()

labour_df = labour_df[labour_df.KM!=10943648]

km_min_val = labour_df.KM.min()
km_max_val = labour_df.KM.max()

km_mean = labour_df.KM.mean()
km_std = labour_df.KM.std()

age_min_val = labour_df.Age_in_Months.min()
age_max_val = labour_df.Age_in_Months.max()

age_mean = labour_df.Age_in_Months.mean()
age_std = labour_df.Age_in_Months.std()

sp = " "+chr(240)+" "

top_n_verbatim=5
top_n_actions=5
top_n_elements=10
top_n_parts=5
page_size=top_n_elements
sim_threshold=0.25

# import fasttext
# ft = fasttext.load_model('C:/Users/ykb2kor/Downloads/cc.en.300.bin/cc.en.300.bin')

def get_top_verbatim(query):

    #try:
    _, _, topic_scores, topic_nums = top2vec_model.search_topics(keywords=query.split(), num_topics=50) # setting high num_topics. because similar topics were merged
    # except Exception as e:
    #     print("Top2vec model OOV error -",str(e))
    #     query_emb = model.encode(query)
    #     _, _, topic_scores, topic_nums = top2vec_model.search_topics_by_vector(vector=query_emb,num_topics=50)
    #     pass

    query_emb = model.encode(query)

    unique_prob = set(query.lower().split()).intersection(diff_prob.keys())

    print("unique_prob",unique_prob)
    
    top_topics = []
    top_verbatims={}
    for i in topic_nums:
        t = replaced_topic_map[i]
        if t not in top_topics and t in df.topic:
            top_topics.append(t)
            
            temp = df[df.topic==t]
            ind = temp.index[0]
            cosim = cosine_similarity(query_emb.reshape(1,-1),verbatims_emb[ind])
            top_index = np.argsort(cosim[0])[::-1][0]

            #print("verbatim_cluster_score",cosim[0][top_index])
            if cosim[0][top_index]>0.3:
                cluster_name = temp.verbatim.iloc[0].split(sp)[top_index]

                if len(unique_prob)==1:
                    unique_prob = list(unique_prob)[0]
                    temp = set(cluster_name.lower().split())
                    common = temp.intersection(diff_prob[unique_prob])
                    if not len(common):
                        top_verbatims[cluster_name] = t
                else:
                    top_verbatims[cluster_name] = t
            
        if len(top_verbatims)>=top_n_verbatim:
            break
        
    return top_verbatims, query_emb

def get_similar_labr_desc_plot(verbatim, action, fil_labour_df):

    labour_desc1 = fil_labour_df.LABR_DESC.unique().tolist()

    labour_desc = []
    for ind,i in enumerate(labour_desc1):
        if "pick" not in i.lower() and 'lamp' not in i.lower():
            labour_desc.append(i)
        
    labour_desc = np.array(labour_desc)
    
    embs = model.encode([action]+labour_desc.tolist())
    action_emb = embs[0]
    labour_desc_emb = embs[1:]

    cosim = cosine_similarity(action_emb.reshape(1,-1),labour_desc_emb)
    
    cond = cosim[0]>sim_threshold

    sorted_index = np.argsort(cosim[0][cond])[::-1][:top_n_elements]

    sim_labours = labour_desc[cond][sorted_index].tolist()

    #print("sim_labours",sim_labours)

    action_words = re.split(r"[^a-zA-Z0-9]",action.lower())

    unique_prob = set(action_words).intersection(diff_prob.keys())

    sim_labours_up = []
    if len(unique_prob)==1:
        unique_prob = list(unique_prob)[0]
        for i in sim_labours:
            temp = set(i.lower().split())
            common = temp.intersection(diff_prob[unique_prob])
            if not len(common):
                sim_labours_up.append(i)
        sim_labours = sim_labours_up

    #removing totally irrelevant labour services reg verbatim

    if len(sim_labours):

        embs = model.encode([verbatim]+sim_labours)
        verbatim_emb = embs[0]
        labours_emb = embs[1:]
    
        cosim = cosine_similarity(verbatim_emb.reshape(1,-1),labours_emb)
        
        cond = cosim[0]>0.2
    
        sim_labours = np.array(sim_labours)[cond]

    labour_dict = dict(zip(sim_labours,cosim[0][cond]*100))

    print("sim labour_dict",labour_dict)

    # labour_output = pd.DataFrame(labour_dict.items(),columns=["Labour","Probablility"])

    t = fil_labour_df[fil_labour_df.LABR_DESC.isin(sim_labours)]

    t = t.groupby(["LABR_DESC"]).size().apply(lambda x:(x/len(t))*100)

    labour_output = t.sort_values(ascending=False).reset_index().rename(columns={"LABR_DESC":"Labour",0:"Prob"})

    labour_output.Prob = labour_output.Prob.apply(lambda x: round(x, 2))
    
    labour_output["code"] = [str(labour_desc2id[i]) for i in labour_output["Labour"]]

    labour_output = labour_output[["code","Labour","Prob"]]

    p = px.bar(labour_output, x='code', y='Prob')

    p.update_layout(xaxis_type='category')

    return p, labour_output


def get_top_actions(topic_id, query_emb, fil_labour_df, verbatim):

    print("Finding top actions for", topic_id)

    topic_action_emb = actions_emb[df[df.topic==topic_id].index[0]]

    sel_index=[]
    actions_list=[]
    scores_list=[]
    for i,(action, score) in enumerate(df[df.topic==topic_id].action_taken.iloc[0]):
        if "found ok" not in action.lower() and "washing" not in action.lower() and (not action.isdigit()) and len(re.sub("[^A-Za-z0-9]", "", action))>2:
            sel_index.append(i)
            actions_list.append(action)
            scores_list.append(score)

    topic_action_emb = topic_action_emb[sel_index]

    cosim = cosine_similarity(query_emb.reshape(1,-1),topic_action_emb)
    top_indexes = np.argsort(cosim[0])[::-1][:top_n_actions]
    
    top_actions = np.array(actions_list)[top_indexes]
    top_scores = np.array(scores_list)[top_indexes]
    
    action_dict_temp = dict(zip(top_actions,top_scores))

    unique_prob = set(verbatim.lower().split()).intersection(diff_prob.keys())

    action_dict = {}
    action2labor_output={}
    for action, score in action_dict_temp.items():
        graph, out_df = get_similar_labr_desc_plot(verbatim, action, fil_labour_df)
        if len(out_df):
            action2labor_output[action] = (graph, out_df)
            if len(unique_prob)==1:
                unique_prob = list(unique_prob)[0]
                temp = set(action.lower().split())
                common = temp.intersection(diff_prob[unique_prob])
                if not len(common):         
                    action_dict[action]=score
            else:
                action_dict[action]=score

    action_dict = dict(sorted(action_dict.items(),key=lambda x:x[1],reverse=True))

    action_df = pd.DataFrame(action_dict.items(),columns = ["Action","Probability"])

    if len(action_df):
        sf = 100/action_df.Probability.sum()
        action_df.Probability = action_df.Probability.apply(lambda x: round(x*sf, 2))

    return action_df, action2labor_output

def dataframe_with_selections(df,column):
    df_with_selections = df.copy()
    #df_with_selections.insert(0, "Select", False)
    event = st.dataframe(df_with_selections, on_select="rerun", selection_mode="single-row", hide_index=True)

    ind = event.selection["rows"][0]

    action = df.iloc[ind][column]

    # edited_df = st.data_editor(
    #     df_with_selections,
    #     hide_index=True,
    #     column_config={"Select": st.column_config.CheckboxColumn(required=True)},
    #     disabled=df.columns,
    # )
    # selected_rows = edited_df[edited_df.Select]
    # return selected_rows.drop('Select', axis=1)
    return action

def get_similar_part_desc_plot(action, selected_labor, fil_part_df, labour_output, verbatim):

    sim_parts = {}

    if len(fil_part_df):

        parts = fil_part_df.PART_DESC.unique()
        parts = np.array(parts)
    
        sim_labour_desc = [selected_labor]

        #sim_labour_desc = [action+" - "+i for i in sim_labour_desc]
    
        embs = model.encode(sim_labour_desc+parts.tolist())

        labr_emb = embs[:len(sim_labour_desc)]
        part_emb = embs[len(sim_labour_desc):]
    
        cosim = cosine_similarity(labr_emb, part_emb)
    
        cond = cosim>0.5
        part2score={}
        for i in range(len(labr_emb)):
            for part,score in zip(parts[cond[i]],cosim[i][cond[i]]):
                if part not in part2score:
                    part2score[part]=score
                elif score>part2score[part]:
                    part2score[part]=score
    
        sorted_parts = sorted(part2score.items(),key=lambda x:x[1],reverse=True)
        
        for part,score in sorted_parts:
            if score>0.7:
                sim_parts[part]=score*100
            elif len(sim_parts)<top_n_parts:
                sim_parts[part]=score*100

    sim_parts = list(sim_parts.keys())

    print("sim_parts",sim_parts)

    #unique_prob = set(action.lower().split()).intersection(diff_prob.keys())

    action_words = re.split(r"[^a-zA-Z0-9]",action.lower())

    unique_prob = set(action_words).intersection(diff_prob.keys())

    sim_parts_up = []
    if len(unique_prob)==1:
        unique_prob = list(unique_prob)[0]
        for i in sim_parts:
            temp = set(i.lower().split())
            common = temp.intersection(diff_prob[unique_prob])
            if not len(common):
                sim_parts_up.append(i)
        sim_parts = sim_parts_up

    #removing totally irrelevant parts reg action

    embs = model.encode([action]+sim_parts)
    action_emb = embs[0]

    if sim_parts:
        sim_parts_emb = embs[1:]
    
        cosim = cosine_similarity(action_emb.reshape(1,-1),sim_parts_emb)
        
        cond = cosim[0]>0.1
    
        sim_parts = np.array(sim_parts)[cond].tolist()

    #adding similar parts wrt action

    cosim = cosine_similarity(action_emb.reshape(1,-1), part_emb)

    cond = cosim[0]>sim_threshold
    
    sim_action_parts = parts[cond]

    #sim_action_parts_dict = dict(zip(sim_action_parts,cosim[0][cond]*100))

    if len(sim_action_parts):

        sim_action_parts_emb = model.encode(sim_action_parts)
    
        cosim = cosine_similarity(sim_action_parts_emb, labr_emb)
    
        indices = np.unique(np.where(cosim>0.2)[0])[:1]
    
        sim_parts = sim_parts+sim_action_parts[indices].tolist()

    t = fil_part_df[fil_part_df.PART_DESC.isin(sim_parts)]

    t = t.groupby(["PART_DESC"]).size().apply(lambda x:(x/len(t))*100)

    parts_output = t.sort_values(ascending=False).reset_index().rename(columns={"PART_DESC":"Part",0:"Prob"})

    #parts_output = pd.DataFrame(sim_parts.items(),columns=["Part","Probablility"])

    parts_output.Prob = parts_output.Prob.apply(lambda x: round(x, 2))

    parts_output["Part_Number"] = [part_desc2id[i] for i in parts_output["Part"]]

    parts_output = parts_output[["Part_Number","Part","Prob"]]
      
    p1 = px.bar(parts_output, x='Part_Number', y='Prob')

    p1.update_layout(xaxis_type='category')

    return p1, parts_output

def slider_sleep():
    time.sleep(1)

query = None
verbatim = None

with st.form(key='input_query'):
    
    with st.sidebar:
        
        query = st.text_input("Query")
        km = st.number_input("KM Driven", min_value=1)
        age = st.number_input("Age in months", min_value=1)
        std =  st.number_input("Standard Deviation", min_value=1, help='Standard deviation unit - Impact on km and age range')
        submit_button = st.form_submit_button(label='Submit')

        inp = max(km_min_val, min(km_max_val,km))
        min_km, max_km = max(km_min_val, int(inp - (2*km_std))), min(km_max_val, int(inp+(std*km_std)))

        inp = max(age_min_val, min(age_max_val,age))
        min_age, max_age = max(age_min_val, int(inp - (2*age_std))), min(age_max_val, int(inp+(std*age_std)))
        
#try:
        
if query:
    
    with st.sidebar:  
        
        print("\nquery",query)

        top_verbatims, query_emb = get_top_verbatim(query)

        #print("top_verbatim",top_verbatims)

        verbatim = st.selectbox(label="Verbatim",options=list(top_verbatims.keys()))

        topic_id = top_verbatims[verbatim]
    
        keys = df[df.topic==topic_id].key.iloc[0].split(sp)
    
        fil_part_df = part_df[part_df.key.isin(keys)]

        fil_labour_df = labour_df[labour_df.key.isin(keys)]
        
        verbatim_service_categories = fil_part_df.SERVC_CATGRY_DESC.unique().tolist() + fil_labour_df.SERVC_TYPE_DESC.unique().tolist()
        verbatim_service_categories = list(set([sevice_category_map[i] for i in verbatim_service_categories if i in sevice_category_map]))
    
        sel_categories = [i for i in verbatim_service_categories if i not in part_rem_serice and i not in labour_rem_service]
        if not sel_categories:
            sel_categories = verbatim_service_categories
    
        # ser_df = pd.DataFrame({"Service category":verbatim_service_categories})
        # event = st.dataframe(ser_df, on_select="rerun", hide_index=True)
        #sel_service_categories = event.selection["rows"]

        sel_service_categories = st.multiselect(
        "Service category",
        options=verbatim_service_categories, 
        default=[i for i in verbatim_service_categories if i in default_service]
            )

        print(sel_service_categories)

        all_sel_sevices = []
        for k in sel_service_categories:
            if k in service_category_groups:
                all_sel_sevices+=service_category_groups[k] 

        print("all_sel_sevices",all_sel_sevices)

        fil_part_df = fil_part_df[fil_part_df.SERVC_CATGRY_DESC.isin(all_sel_sevices)]
        fil_labour_df = fil_labour_df[fil_labour_df.SERVC_TYPE_DESC.isin(all_sel_sevices)]

        km_range = st.slider(
        "KM range",
        min_km, max_km, (min_km,max_km), disabled=True, help = "The output range extends from a value that is one standard deviation below the input to a value that is one standard deviation above the input")

        st.write("Selected KM Range",km_range)

        print("KM Range",km_range)

        fil_labour_df = fil_labour_df[(fil_labour_df.KM >= min_km) & (fil_labour_df.KM <= max_km)]
        keys = fil_labour_df.key.tolist()
        fil_part_df = fil_part_df[fil_part_df.key.isin(keys)]
            
        age_range = st.slider(
        "Age range in months",
        min_age, max_age, (min_age,max_age), disabled=True, help = "The output range extends from a value that is one standard deviation below the input to a value that is one standard deviation above the input.")

        st.write("Selected Age Range",age_range)

        print("Age Range",age_range)

        fil_labour_df = fil_labour_df[(fil_labour_df.Age_in_Months >= min_age) & (fil_labour_df.Age_in_Months <= max_age)]
        keys = fil_labour_df.key.tolist()
        fil_part_df = fil_part_df[fil_part_df.key.isin(keys)]

        choices = ["XUV700"] 
        car_model = st.selectbox(label="Model",options=choices)

if verbatim:

    st.subheader("Actions")
    action_df, action2labor_output = get_top_actions(topic_id, query_emb, fil_labour_df, verbatim)
    #action_df = dataframe_with_selections(action_df)
    #st.dataframe(selected_action)

    selected_action = None
    try:
        #selected_action = action_df.Action.iloc[0]
        selected_action = dataframe_with_selections(action_df,"Action")
        print("selected_action",selected_action)
    except:
        pass

    if selected_action: 
    
        if len(fil_part_df) or len(fil_labour_df):
    
            
        
            # if model!="All Models":
            #     choices, p1, p2, df1, df2 = model_out(verbatim,car_model)
        
            out11, out12 = st.columns([0.5,0.5])
        
            with st.container():
    
                if  len(fil_labour_df):
    
                    #p1, df1 = get_similar_labr_desc_plot(verbatim, selected_action, fil_labour_df)
                    p1, df1 = action2labor_output[selected_action]
        
                    with out11:
                        out11.subheader("Labour Services")

                        selected_labor = None
                        try:
                            selected_labor = dataframe_with_selections(df1,"Labour")
                            print("selected_labor",selected_labor)
                        except:
                            pass
            
                    with out12:
                        
                        if selected_labor:

                            p2, df2 = get_similar_part_desc_plot(selected_action, selected_labor, fil_part_df, df1, verbatim)

                            out12.subheader("Parts")
                            st.dataframe(df2)
    
                else:
                    st.subheader("No Labour services/ Parts data under selected filters")
                    
        else:
            with st.container():
                st.subheader("No Part and Labour service data under selected filters")
    
# except Exception as e:
#     print("Exception", str(e))
#     st.subheader("The query words were not recognized by the model. Please try using different query words.")

