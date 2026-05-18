from flask import Blueprint, render_template, current_app, abort, redirect, url_for
from pathlib import Path

from src.analytics.deep_process_analytics import DeepProcessAnalytics
from src.analytics.wafer_sort_analytics import WaferSortAnalytics
from src.analytics.packaged_process_analytics import PackagedProcessAnalytics

# Note the url_prefix matches your sidebar links
dashboard_bp = Blueprint("dashboard_bp", __name__)


def resolve_data_path(filename: str) -> str:
    base_dir = Path(current_app.config.get("DATA_DIR", "src/data"))
    resolved = base_dir / "raw_data" / filename
    if not resolved.exists():
        current_app.logger.error(f"Missing mandatory manufacturing data matrix: {resolved}")
        abort(404, description="Manufacturing database file missing.")
    return str(resolved)


@dashboard_bp.route('/')
def index():
    """
    Root URL for the optimization suite panel.
    Redirects directly to the wafer sort dashboard by default.
    """
    return redirect(url_for('dashboard_bp.wafer_sort_view'))


@dashboard_bp.route('/sort-layer')
def wafer_sort_view():
    csv_path = resolve_data_path("silicon_packaged_data_groups.csv")
    print(f"DEBUG: csv_path: {csv_path}")
    analytics = WaferSortAnalytics(csv_path)

    return render_template(
        'wafer_dashboard.html',
        stats=analytics.get_summary_stats(),
        spatial_clusters=analytics.generate_plot('spatial_clusters'),
        virtual_metrology=analytics.generate_plot('virtual_metrology'),
        root_cause=analytics.generate_plot('root_cause')
    )


@dashboard_bp.route('/package-layer')
def package_view():
    csv_path = resolve_data_path("silicon_packaged_data_groups.csv")
    analytics = PackagedProcessAnalytics(csv_path)

    return render_template(
        'package_dashboard.html',
        stats=analytics.get_summary_stats(),
        env_plot=analytics.generate_plot('environmental'),
        serdes_plot=analytics.generate_plot('serdes'),
        corner_plot=analytics.generate_plot('process_corners'),
        reliability_plot=analytics.generate_plot('reliability')
    )


@dashboard_bp.route('/deep-analytics')
def deep_analytics_view():
    csv_path = resolve_data_path("silicon_packaged_data_groups.csv")
    analytics = DeepProcessAnalytics(csv_path)

    return render_template(
        'deep_dashboard.html',
        stats=analytics.get_summary_stats(),
        ae_plot=analytics.generate_plot('autoencoder_loss'),
        family_plot=analytics.generate_plot('silicon_families')
    )
