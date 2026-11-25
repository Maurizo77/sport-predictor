def get_ucl_fixtures(target_date=None):
    df = pd.DataFrame(UCL_FIXTURES)
    if target_date:
        df = df[df["Date"] == target_date].copy()
    return df