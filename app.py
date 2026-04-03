import streamlit as st
import pandas as pd
from pipeline import preprocess_dataset, run_pipeline, format_results_table

st.title("RL Feature Selection System")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.write("Preview:", df.head())

    target = st.selectbox("Select Target Column", df.columns)
    sample_option = st.selectbox("Select sample size", ["5000", "10000", "20000", "Full dataset"])
    use_rl = st.checkbox("Enable RL Feature Selection", value=True)

    if st.button("Run Model"):
        with st.spinner("Processing ... this may take time ⏳"):
            if sample_option != "Full dataset":
                size = int(sample_option)
                if len(df) > size:
                    df = df.groupby(target, group_keys=False).apply(
                        lambda x: x.sample(min(len(x), size // 2), random_state=42)
                    ).reset_index(drop=True)
                    st.warning(f"Using BALANCED sample of {len(df)} rows")
        
            X, y = preprocess_dataset(df, target)

            # Pass our simplified flags
            all_results = run_pipeline(X, y, use_rl=use_rl)

            # Display Baseline
            st.subheader("🚀 Baseline")
            st.write(f"G-Mean: {all_results['baseline']['gmean']:.4f}")
            st.write(f"Time: {all_results['baseline']['runtime']}s")

            # Display RL Only if enabled
            if "rl_only" in all_results:
                st.subheader("🧠 RL Feature Selection")
                st.write(f"G-Mean: {all_results['rl_only']['gmean']:.4f}")
                st.write(f"Time: {all_results['rl_only']['runtime']}s")
                
                # Show comparison
                diff = all_results['rl_only']['gmean'] - all_results['baseline']['gmean']
                st.metric("G-Mean Improvement", f"{diff:.4f}")

            

            table = format_results_table(all_results)
            st.subheader("Model Comparison Results:")
            st.dataframe(table)

            st.json(all_results)