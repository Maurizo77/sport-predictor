import os

import pandas as pd
import streamlit as st

from team_names import full


def show_basketball_picks(base):
    log_path = os.path.join(base, "predictions_log.csv")
    if not os.path.exists(log_path):
        st.info("No predictions saved yet.")
        return

    log = pd.read_csv(log_path, dtype={"GAME_ID": str})
    latest = log[log["DATE"] == log["DATE"].max()].copy()

    injuries = None
    injuries_path = os.path.join(base, "injuries_log.csv")
    if os.path.exists(injuries_path):
        injuries = pd.read_csv(injuries_path)
        if injuries.empty:
            injuries = None

    latest["Game"] = latest["AWAY"].map(full) + " @ " + latest["HOME"].map(full)
    latest["Pick"] = latest["PICK"].map(full)
    latest["Confidence %"] = (
        latest["HOME_WIN_PROB"].apply(lambda p: max(p, 1 - p)) * 100
    ).round(0)
    latest["Predicted total"] = latest["PRED_TOTAL"]
    columns = ["Game", "Pick", "Confidence %", "Predicted total"]

    st.caption("Latest saved predictions, for " + str(log["DATE"].max()))
    st.dataframe(
        latest[columns],
        hide_index=True,
        use_container_width=True,
        column_config={
            "Confidence %": st.column_config.NumberColumn(format="%.0f%%"),
            "Predicted total": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    if injuries is not None:
        st.subheader("Injury report for these teams")
        st.caption(
            "Source: ESPN's free injury feed (unofficial), fetched "
            + str(injuries["FETCHED"].iloc[0])
            + ". A heads-up, not a guarantee: statuses can lag behind the latest news, "
            "and the list doesn't say who the key players are."
        )
        names = set(latest["AWAY"].map(full)) | set(latest["HOME"].map(full))
        view = injuries[injuries["TEAM"].isin(names)].sort_values(["TEAM", "STATUS"])
        with st.expander("Injury report details"):
            st.dataframe(
                view[["TEAM", "PLAYER", "POSITION", "STATUS", "PROBLEM", "RETURN", "NOTE"]],
                hide_index=True,
                use_container_width=True,
            )
