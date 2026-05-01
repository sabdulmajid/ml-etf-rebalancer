from portfolio.research import ResearchConfig, build_research_artifacts

def run_pipeline():
    """Build the research artifacts consumed by the dashboard."""
    print("----- Building ETF sector rotation research artifacts -----")
    result = build_research_artifacts(ResearchConfig())
    print(f"Artifacts written to {result['artifact_dir']}")
    print("\nCurrent allocation:")
    for _, row in result["current_allocation"].iterrows():
        print(f"  {row['ticker']} ({row['sector']}): {row['weight']:.2%}")
    print("\nWalk-forward performance:")
    print(result["metrics"][["Total Return", "CAGR", "Sharpe Ratio", "Max Drawdown"]])
    print("----- Pipeline completed successfully -----")
    return result

if __name__ == "__main__":
    run_pipeline()
