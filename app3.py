import streamlit as st
import pandas as pd
from pipeline import preprocess_dataset, run_pipeline

st.title("RL-MFGWfs Fraud Detection System")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.write("Preview:", df.head())

    target = st.selectbox("Select Target Column", df.columns)

    sample_option = st.selectbox(
        "Select sample size",
        ["5000", "10000", "20000", "Full dataset"]
        #["10000", "20000", "50000", "Full dataset"]
    )

    use_rl = st.checkbox("Enable RL Feature Grouping", value=True)

    if st.button("Run Model"):
        with st.spinner("Processing... this may take time ⏳"):

            # Sampling
            if sample_option != "Full dataset":
                size = int(sample_option)

                if len(df) > size:
                    df = df.groupby(target).apply(
                        lambda x: x.sample(min(len(x), size // 2), random_state=42)
                    ).reset_index(drop=True)

                    st.warning(f"Using BALANCED sample of {size} rows")
        
            X, y = preprocess_dataset(df, target)

            results = run_pipeline(X, y, use_rl=use_rl)

            st.success("Done!")

            st.write("### Results")
            st.json(results)

            '''
            if sample_option != "Full dataset":
                size = int(sample_option)
                if len(df) > size:
                    df = df.sample(size, random_state=42)
                    st.warning(f"Using sample of {size} rows")
        '''