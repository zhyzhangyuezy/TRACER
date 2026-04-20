from __future__ import annotations

from pathlib import Path

from atlasv2_public_results import (
    build_ablation_table_tex,
    build_ait_ads_benchmark_stats_table_tex,
    build_ait_ads_discrimination_calibration_table_tex,
    build_ait_ads_results_table_tex,
    build_atlas_raw_discrimination_calibration_table_tex,
    build_primary_discrimination_calibration_table_tex,
    build_event_extended_table_tex,
    build_event_retrieval_table_tex,
    build_primary_operating_point_table_tex,
    build_raw_public_benchmark_stats_table_tex,
    build_raw_public_results_table_tex,
    build_main_table_tex,
    build_public_benchmark_stats_table_tex,
    build_robustness_table_tex,
    build_workbook_probe_stats_table_tex,
    write_summary_files,
)


def main() -> None:
    figure_dir = Path(__file__).resolve().parent
    summary_json, summary_csv = write_summary_files()
    (figure_dir / "tab_public_benchmark_stats.tex").write_text(build_public_benchmark_stats_table_tex(), encoding="utf-8")
    (figure_dir / "tab_ait_ads_benchmark_stats.tex").write_text(build_ait_ads_benchmark_stats_table_tex(), encoding="utf-8")
    (figure_dir / "tab_ait_ads_results.tex").write_text(build_ait_ads_results_table_tex(), encoding="utf-8")
    (figure_dir / "tab_atlas_raw_benchmark_stats.tex").write_text(build_raw_public_benchmark_stats_table_tex(), encoding="utf-8")
    (figure_dir / "tab_atlas_raw_results.tex").write_text(build_raw_public_results_table_tex(), encoding="utf-8")
    (figure_dir / "tab_public_main_results.tex").write_text(build_main_table_tex(), encoding="utf-8")
    (figure_dir / "tab_public_operating_point_metrics.tex").write_text(build_primary_operating_point_table_tex(), encoding="utf-8")
    (figure_dir / "tab_public_discrimination_calibration.tex").write_text(build_primary_discrimination_calibration_table_tex(), encoding="utf-8")
    (figure_dir / "tab_ait_ads_discrimination_calibration.tex").write_text(build_ait_ads_discrimination_calibration_table_tex(), encoding="utf-8")
    (figure_dir / "tab_atlas_raw_discrimination_calibration.tex").write_text(build_atlas_raw_discrimination_calibration_table_tex(), encoding="utf-8")
    (figure_dir / "tab_public_event_extended.tex").write_text(build_event_extended_table_tex(), encoding="utf-8")
    (figure_dir / "tab_public_ablations.tex").write_text(build_ablation_table_tex(), encoding="utf-8")
    (figure_dir / "tab_public_event_retrieval.tex").write_text(build_event_retrieval_table_tex(), encoding="utf-8")
    (figure_dir / "tab_workbook_probe_stats.tex").write_text(build_workbook_probe_stats_table_tex(), encoding="utf-8")
    (figure_dir / "tab_public_robustness.tex").write_text(build_robustness_table_tex(), encoding="utf-8")
    latex_includes = "\n".join(
        [
            "% Figures",
            r"\begin{figure*}[t]",
            r"    \centering",
            r"    \includegraphics[width=\textwidth]{figures/f1_overview.png}",
            r"    \caption{Updated overview of the TRACER pipeline. An alert prefix is decomposed into slow trend and burst-sensitive residual views, while a train-only retrieval path queries historical analogs. The core first forms a retrieval-backed base score, then combines trend/residual forecasting, route selection, and bounded calibration to produce the final warning. The lower branch summarizes the training path with auxiliary-horizon, calibration, and future-signature contrastive objectives.}",
            r"    \label{fig:tracer-overview}",
            r"\end{figure*}",
            "",
            r"\begin{figure*}[t]",
            r"    \centering",
            r"    \includegraphics[width=\textwidth]{figures/fig_public_main_comparison.pdf}",
            r"    \caption{Chronological comparison on the public ATLASv2 benchmark.}",
            r"    \label{fig:public-main-comparison}",
            r"\end{figure*}",
            "",
            r"\begin{figure*}[t]",
            r"    \centering",
            r"    \includegraphics[width=\textwidth]{figures/fig_public_event_disjoint.pdf}",
            r"    \caption{Family-held-out event-disjoint comparison on the public ATLASv2 benchmark.}",
            r"    \label{fig:public-event-disjoint}",
            r"\end{figure*}",
            "",
            r"\begin{figure*}[t]",
            r"    \centering",
            r"    \includegraphics[width=\textwidth]{figures/fig_public_ablations.pdf}",
            r"    \caption{Novelty-isolation and ablation analysis on the public ATLASv2 benchmark.}",
            r"    \label{fig:public-ablations}",
            r"\end{figure*}",
            "",
            r"\begin{figure*}[t]",
            r"    \centering",
            r"    \includegraphics[width=\textwidth]{figures/fig_public_tradeoff_frontier.pdf}",
            r"    \caption{Trade-offs between forecasting accuracy, lead time, and analog quality on the public benchmark.}",
            r"    \label{fig:public-tradeoff-frontier}",
            r"\end{figure*}",
            "",
            r"\begin{figure*}[t]",
            r"    \centering",
            r"    \includegraphics[width=\textwidth]{figures/fig_public_qualitative_cases.pdf}",
            r"    \caption{Qualitative diagnosis of retrieval behavior on the public ATLASv2 benchmark. The figure is intentionally limitation-oriented: the positive case is rare, and several cases share the same top-1 analog across methods.}",
            r"    \label{fig:public-qualitative-cases}",
            r"\end{figure*}",
            "",
            "% Tables",
            r"\input{figures/tab_public_benchmark_stats.tex}",
            r"\input{figures/tab_public_main_results.tex}",
            r"\input{figures/tab_public_ablations.tex}",
            r"\input{figures/tab_workbook_probe_stats.tex}",
            r"\input{figures/tab_public_robustness.tex}",
            "",
            f"% Summary data: {summary_json.name}, {summary_csv.name}",
        ]
    )
    (figure_dir / "latex_includes.tex").write_text(latex_includes + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
