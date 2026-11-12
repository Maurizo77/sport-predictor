import os

import pandas as pd
import streamlit as st


def confidence_table(res):
    bins = [0.5, 0.6, 0.7, 0.8, 1.01]
    labels = ["50-59%", "60-69%", "70-79%", "80% and above"]
    data = res.dropna(subset=["CONFIDENCE"]).copy()
    data["Confidence band"] = pd.cut(
        data["CONFIDENCE"], bins=bins, labels=labels, right=False
    )
    table = (
        data.groupby("Confidence band", observed=True)["CORRECT"]
        .agg(["count", "sum"])
        .reset_index()
    )
    table.columns = ["Confidence band", "Games", "Correct"]
    table["Confidence band"] = table["Confidence band"].astype(str)
    table["Accuracy"] = (
        (table["Correct"] / table["Games"] * 100).round(0).astype(int).astype(str) + "%"
    )
    return table


def show_results(base):
    results_path = os.path.join(base, "results_log.csv")
    if not os.path.exists(results_path):
        st.info("No checked results yet.")
        return

    res = pd.read_csv(results_path, dtype={"GAME_ID": str})
    res["Result"] = res["CORRECT"].map({True: "✔", False: "✘"})
    res["Total error"] = (res["PRED_TOTAL"] - res["REAL_TOTAL"]).abs().round(1)

    c1, c2, c3 = st.columns(3)
    c1.metric("Games checked", len(res))
    c2.metric("Correct", int(res["CORRECT"].sum()))
    c3.metric("Accuracy", str(round(res["CORRECT"].mean() * 100)) + "%")
    st.metric("Average total error (points)", round(res["Total error"].mean(), 1))

    columns = ["DATE", "GAME", "PICK", "WINNER", "Result"]
    if "CONFIDENCE" in res.columns and res["CONFIDENCE"].notna().any():
        st.subheader("Accuracy by confidence")
        st.dataframe(confidence_table(res), hide_index=True)
        st.caption(
            "How often the picks in each confidence band were right. "
            "Small groups are unreliable, so it needs many games."
        )
        res["Confidence %"] = (res["CONFIDENCE"] * 100).round(0)
        columns.append("Confidence %")

    columns += ["PRED_TOTAL", "REAL_TOTAL", "Total error"]
    st.subheader("All checked games")
    st.dataframe(res[columns], hide_index=True)