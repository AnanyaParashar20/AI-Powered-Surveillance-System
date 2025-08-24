# FILE: src/dashboard_app.py

"""
Surveillance Dashboard - Security monitoring and event analysis
"""

import streamlit as st
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import cv2
import numpy as np
from PIL import Image


def load_events_data(file_path):
    """Load and process events data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        
        # Standardize column names for compatibility
        if 'type' in df.columns and 'event_type' not in df.columns:
            df['event_type'] = df['type']
        if 'time_sec' in df.columns and 'timestamp' not in df.columns:
            # Convert time_sec to timestamp for better display
            df['timestamp'] = pd.to_datetime(df['time_sec'], unit='s')
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        if 'score' in df.columns and 'confidence' not in df.columns:
            df['confidence'] = df['score']
        
        st.sidebar.success(f"Loaded {len(df)} events from {os.path.basename(file_path)}")
        return df
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")
        return pd.DataFrame()

def plot_timeline(df):
    """Create premium timeline visualization with linear seconds axis"""
    if df.empty:
        st.markdown("""
        <div class="timeline-container">
            <h3 class="section-title">Event Timeline</h3>
            <p style="color: #7f8c8d;">No data available for timeline visualization</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Use compatible column names
    timestamp_col = 'timestamp' if 'timestamp' in df.columns else None
    event_type_col = 'event_type' if 'event_type' in df.columns else 'type'
    confidence_col = 'confidence' if 'confidence' in df.columns else 'score'
    
    # Convert timestamps to seconds from start for linear axis
    df_sorted = df.sort_values(timestamp_col if timestamp_col else 'time_sec')
    
    if timestamp_col:
        start_time = df_sorted[timestamp_col].min()
        df_sorted['seconds_from_start'] = (df_sorted[timestamp_col] - start_time).dt.total_seconds()
    else:
        # Use time_sec directly
        start_time = df_sorted['time_sec'].min()
        df_sorted['seconds_from_start'] = df_sorted['time_sec'] - start_time
    
    # Create premium Plotly timeline
    fig = go.Figure()
    
    # Red theme color scheme for event types
    colors = {
        'abandonment': '#dc2626',     # Dark red
        'loitering': '#f97316',       # Orange-red  
        'unusual': '#7c2d12'          # Dark red-brown
    }
    
    # Event type symbols
    symbols = {
        'abandonment': 'diamond',
        'loitering': 'circle',
        'unusual': 'triangle-up'
    }
    
    for event_type in df_sorted[event_type_col].unique():
        type_data = df_sorted[df_sorted[event_type_col] == event_type]
        
        # Prepare hover data
        if timestamp_col:
            time_display = type_data[timestamp_col].dt.strftime('%H:%M:%S')
        else:
            time_display = type_data['time_sec'].astype(str) + 's'
        
        video_id = type_data.get('video_id', type_data.get('track_id', 'N/A'))
        
        fig.add_trace(go.Scatter(
            x=type_data['seconds_from_start'],
            y=[event_type] * len(type_data),
            mode='markers',
            marker=dict(
                size=12,
                color=colors.get(event_type, '#3498db'),
                symbol=symbols.get(event_type, 'circle'),
                line=dict(width=2, color='white')
            ),
            name=f"{event_type.title()} ({len(type_data)})",
            hovertemplate=(
                f"<b>{event_type.title()} Event</b><br>"
                "Time: %{customdata[0]}<br>"
                "Confidence: %{customdata[1]:.1%}<br>"
                "ID: %{customdata[2]}<br>"
                "<extra></extra>"
            ),
            customdata=list(zip(
                time_display,
                type_data[confidence_col],
                video_id
            ))
        ))
    
    # Premium styling with red theme
    fig.update_layout(
        title={
            'text': 'Security Event Timeline',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#000000', 'family': 'Segoe UI, sans-serif'}
        },
        xaxis_title="Time (seconds from start)",
        yaxis_title="Event Type",
        plot_bgcolor='rgba(254, 242, 242, 0.5)',
        paper_bgcolor='white',
        font=dict(family="Segoe UI, sans-serif", size=12, color="#000000"),
        hovermode='closest',
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(220,38,38,0.3)",
            borderwidth=2
        )
    )
    
    # Style axes with red theme
    fig.update_xaxes(
        gridcolor='rgba(220, 38, 38, 0.2)',
        showgrid=True,
        zeroline=False,
        tickformat='.0f',
        tickfont=dict(color='#000000'),
        title=dict(font=dict(color='#000000'))
    )
    
    fig.update_yaxes(
        gridcolor='rgba(220, 38, 38, 0.2)',
        showgrid=True,
        zeroline=False,
        tickfont=dict(color='#000000'),
        title=dict(font=dict(color='#000000'))
    )
    
    # Display in clean container
    st.markdown("""
    <div class="timeline-container">
        <h3 class="section-title">Event Timeline Analysis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Timeline statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Duration", f"{df_sorted['seconds_from_start'].max():.0f}s")
    
    with col2:
        # Events per minute, guard divide-by-zero
        total_seconds = max(1.0, float(df_sorted['seconds_from_start'].max()))
        events_per_min = len(df) / (total_seconds / 60.0)
        st.metric("Event Density", f"{events_per_min:.1f}/min")
    
    with col3:
        # Guard for empty mode
        if event_type_col in df.columns and not df[event_type_col].dropna().empty:
            try:
                most_common = df[event_type_col].mode().iloc[0]
                st.metric("Most Common", str(most_common).title())
            except Exception:
                st.metric("Most Common", "N/A")
        else:
            st.metric("Most Common", "N/A")

def display_alert_summary(df: pd.DataFrame):
    """Display event summary metrics."""
    # Simple metrics display
    st.subheader("📊 Event Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Alerts", len(df))
    
    with col2:
        unique_types = df['event_type'].nunique() if 'event_type' in df.columns else df['type'].nunique()
        st.metric("Event Types", unique_types)
    
    with col3:
        # Safely compute average confidence — handle missing column or empty values
        confidence_col = 'confidence' if 'confidence' in df.columns else ('score' if 'score' in df.columns else None)
        if confidence_col is not None and not df[confidence_col].dropna().empty:
            avg_confidence = float(df[confidence_col].mean())
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        else:
            st.metric("Avg Confidence", "N/A")
    
    with col4:
        if 'timestamp' in df.columns:
            duration = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
        else:
            duration = df['time_sec'].max() - df['time_sec'].min()
        st.metric("Duration", f"{duration:.0f}s")

def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply filters to the dataframe."""
    filtered_df = df.copy()
    
    # Event type filter
    event_type_col = 'event_type' if 'event_type' in df.columns else 'type'
    if filters.get('event_types') and event_type_col in df.columns:
        filtered_df = filtered_df[filtered_df[event_type_col].isin(filters['event_types'])]
    
    # Score range filter
    score_col = 'confidence' if 'confidence' in df.columns else 'score'
    if score_col in df.columns and filters.get('score_range'):
        filtered_df = filtered_df[
            (filtered_df[score_col] >= filters['score_range'][0]) &
            (filtered_df[score_col] <= filters['score_range'][1])
        ]
    
    # Time range filter
    time_col = 'timestamp' if 'timestamp' in df.columns else 'time_sec'
    if time_col in df.columns and filters.get('time_range'):
        if time_col == 'timestamp':
            # Convert seconds to timestamp for filtering
            start_ts = df[time_col].min() + pd.Timedelta(seconds=filters['time_range'][0])
            end_ts = df[time_col].min() + pd.Timedelta(seconds=filters['time_range'][1])
            filtered_df = filtered_df[
                (filtered_df[time_col] >= start_ts) &
                (filtered_df[time_col] <= end_ts)
            ]
        else:
            filtered_df = filtered_df[
                (filtered_df[time_col] >= filters['time_range'][0]) &
                (filtered_df[time_col] <= filters['time_range'][1])
            ]
    
    # Event data display
    st.write(f"📊 Showing {len(filtered_df)} of {len(df)} events")
    
    if not filtered_df.empty:
        # Select columns to display
        display_cols = []
        if 'timestamp' in filtered_df.columns:
            display_cols.append('timestamp')
        elif 'time_sec' in filtered_df.columns:
            display_cols.append('time_sec')
        
        if 'event_type' in filtered_df.columns:
            display_cols.append('event_type')
        elif 'type' in filtered_df.columns:
            display_cols.append('type')
            
        if 'confidence' in filtered_df.columns:
            display_cols.append('confidence')
        elif 'score' in filtered_df.columns:
            display_cols.append('score')
            
        if 'frame_idx' in filtered_df.columns:
            display_cols.append('frame_idx')
        if 'track_id' in filtered_df.columns:
            display_cols.append('track_id')
        if 'image_path' in filtered_df.columns:
            display_cols.append('image_path')
        
        st.dataframe(filtered_df[display_cols], use_container_width=True)
        
        # Simple event type counts
        event_type_col = 'event_type' if 'event_type' in filtered_df.columns else 'type'
        if event_type_col in filtered_df.columns:
            st.subheader("Event Type Distribution")
            event_counts = filtered_df[event_type_col].value_counts()
            st.bar_chart(event_counts)
    else:
        st.info("No events match your current filters. Try adjusting the filter criteria.")
    
    return filtered_df

def display_alert_images(df: pd.DataFrame, image_dir: str):
    """Display alert images with color-coded event types."""
    if df.empty or 'image_path' not in df.columns:
        st.info("💡 No image path column found in data")
        return
    
    st.subheader("📸 Alert Images")
    
    # Filter for alerts with images
    df_with_images = df[df['image_path'].notna() & (df['image_path'] != '')]
    
    if df_with_images.empty:
        st.info("No alert images available in the selected data")
        return
    
    # Check how many images actually exist
    existing_images = []
    missing_images = []
    
    for idx, alert in df_with_images.iterrows():
        image_path = alert['image_path']
        
        # Handle both absolute and relative paths
        if not os.path.isabs(image_path):
            full_path = os.path.join(os.getcwd(), image_path)
        else:
            full_path = image_path
        
        if os.path.exists(full_path):
            existing_images.append((idx, alert, full_path))
        else:
            missing_images.append(image_path)
    
    # Show status
    st.write(f"📊 Found {len(df_with_images)} image references:")
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"✅ {len(existing_images)} images available")
    with col2:
        if missing_images:
            st.warning(f"⚠️ {len(missing_images)} images missing")
    
    # Show missing image info (kept as in your version)
    if missing_images and st.expander("Show missing image paths", expanded=False):
        for missing_path in missing_images[:5]:  # Show first 5
            st.text(f"❌ {missing_path}")
        if len(missing_images) > 5:
            st.text(f"... and {len(missing_images) - 5} more")
    
    if not existing_images:
        st.error("❌ No valid images found. This may be demo data referencing non-existent images.")
        st.info("💡 Try selecting a different CSV file that has actual processed video data.")
        return
    
    # Create image grid for existing images
    cols_per_row = 3
    rows = (len(existing_images) + cols_per_row - 1) // cols_per_row
    
    for row in range(rows):
        cols = st.columns(cols_per_row)
        
        for col_idx in range(cols_per_row):
            alert_idx = row * cols_per_row + col_idx
            if alert_idx >= len(existing_images):
                break
            
            idx, alert, full_path = existing_images[alert_idx]
            
            with cols[col_idx]:
                try:
                    image = Image.open(full_path)
                    
                    # Get event info
                    event_type = alert.get('event_type', alert.get('type', 'unknown'))
                    confidence = alert.get('confidence', alert.get('score', 0))
                    frame_info = alert.get('frame_idx', alert.get('time_sec', 'N/A'))
                    
                    # Color-coded caption based on event type
                    if event_type == 'abandonment':
                        color = "🔴"  # Red
                    elif event_type == 'loitering':
                        color = "🟡"  # Yellow
                    elif event_type == 'unusual':
                        color = "🟣"  # Purple
                    else:
                        color = "🔵"  # Blue
                    
                    caption = f"{color} Frame {frame_info}: {event_type.upper()} (Score: {confidence:.1%})"
                    
                    st.image(image, caption=caption, width=300)
                    
                except Exception as e:
                    st.error(f"Failed to load image {os.path.basename(full_path)}: {e}")


def main():
    """Surveillance Analytics Dashboard."""
    # Page Configuration
    st.set_page_config(
        page_title="Surveillance Analytics",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # -------------------- THEME + UPDATED SIDEBAR CSS (NO GREY) --------------------
    st.markdown("""
    <style>
    /* ===== MAIN AREA ===== */
    [data-testid="stAppViewContainer"] { background:#fff; }
    .main > div { padding-top:1rem; background:#fff; }
    
    .main-header{
      background:linear-gradient(135deg,#dc2626 0%,#ef4444 100%);
      padding:1.5rem; border-radius:8px; margin-bottom:1.5rem;
      border-left:4px solid #b91c1c; box-shadow:0 2px 4px rgba(220,38,38,.2);
    }
    /* Force all main area text black for readability */
    .main-title{ color:#000000!important; font-size:1.8rem; font-weight:600; margin:0; font-family:'Segoe UI',sans-serif; }
    .main-subtitle{ color:#000000!important; font-size:.95rem; margin:.3rem 0 0; font-weight:400; }
    
    .data-section,.timeline-container{
      background:#fff; padding:1.5rem; border-radius:6px; border:2px solid #ef4444;
      margin:1rem 0; box-shadow:0 1px 3px rgba(220,38,38,.1);
    }
    .section-title{ color:#111827!important; font-size:1.1rem; font-weight:600; margin-bottom:1rem; border-bottom:2px solid #dc2626; padding-bottom:.5rem; }
    .status-info{ background:#fef2f2; border-left:4px solid #dc2626; padding:.8rem; border-radius:4px; color:#111827; margin:1rem 0; }
    .block-container{ padding-top:1rem; padding-bottom:1rem; }
    
    [data-testid="metric-container"]{
      background:#fff; border:1px solid #ef4444; padding:1rem; border-radius:6px; box-shadow:0 1px 3px rgba(220,38,38,.1);
    }

        /* Make all standard textual elements in main content black */
        .main-header *, .data-section, .data-section *, .timeline-container, .timeline-container *,
        .status-info, .status-info *, .block-container, .block-container *,
        h1, h2, h3, h4, h5, h6, p, span, label, div, li, strong, em {
            color:#000000 !important;
            text-shadow:none !important;
        }
    
    /* ===== CONTROL PANEL / SIDEBAR (no grey) ===== */
    section[data-testid="stSidebar"]{
      background:#ffffff!important;
    }
    section[data-testid="stSidebar"] > div{
      background:linear-gradient(180deg,#ffffff 0%,#fff5f5 60%,#ffffff 100%)!important;
      border-right:3px solid #dc2626!important;
      box-shadow:2px 0 8px rgba(0,0,0,.06)!important;
      padding-top:.5rem!important;
    }
    /* Sidebar headings & text */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] .stMarkdown{
      color:#111827!important;
    }
    /* Badge-style title */
    section[data-testid="stSidebar"] .cp-title{
      display:block; background:#fff; color:#111827;
      border:2px solid #111827; border-radius:8px; padding:.5rem .75rem;
      font-weight:800; letter-spacing:.5px; text-transform:uppercase; text-align:center;
      box-shadow:0 2px 4px rgba(0,0,0,.06);
    }
    /* Inputs */
    section[data-testid="stSidebar"] .stRadio > label,
    section[data-testid="stSidebar"] .stSelectbox > div > div,
    section[data-testid="stSidebar"] .stMultiSelect > div > div,
    section[data-testid="stSidebar"] .stTextInput > div > div > input,
    section[data-testid="stSidebar"] .stFileUploader,
    section[data-testid="stSidebar"] .stSlider{
      background:#fff!important; border:2px solid #111827!important; border-radius:8px!important; color:#111827!important;
    }
    section[data-testid="stSidebar"] .stRadio > label:hover{
      background:#f3f4f6!important; border-color:#111827!important; transform:translateX(2px);
    }
    /* Buttons */
    section[data-testid="stSidebar"] .stButton > button{
      background:#fff!important; color:#111827!important; border:2px solid #111827!important;
      border-radius:8px!important; font-weight:700!important; text-transform:uppercase; letter-spacing:.5px; transition:.2s;
    }
    section[data-testid="stSidebar"] .stButton > button:hover{
      background:#111827!important; color:#fff!important; border-color:#111827!important;
    }
    /* Slider accents */
    section[data-testid="stSidebar"] .stSlider [data-testid="stTickBar"]{
      background:linear-gradient(90deg,#dc2626,#ef4444)!important;
    }
    /* Divider */
    section[data-testid="stSidebar"] hr{
      border:none!important; height:2px!important; background:#111827!important; margin:1rem 0!important;
    }
    /* Keep main content inheriting its colors */
    [data-testid="stAppViewContainer"] *{ color:inherit; }
    </style>
    """, unsafe_allow_html=True)
    # -------------------------------------------------------------------------------
    
    # Professional Header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">Surveillance Analytics Dashboard</h1>
        <p class="main-subtitle">Security monitoring and event analysis system</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Sidebar Navigation (new badge title)
    st.sidebar.markdown('<span class="cp-title">🎛️ Control Panel</span>', unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    # Data Source Section
    st.sidebar.markdown("### 📊 Data Source")
    data_source = st.sidebar.radio("Select Input:", ["Auto-load Demo", "Upload CSV", "File Path"])
    
    df = pd.DataFrame()
    
    if data_source == "Auto-load Demo":
        # Look for demo files
        demo_files = []
        output_dir = "data/output"
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                if file.endswith('.csv'):
                    demo_files.append(os.path.join(output_dir, file))
        
        if demo_files:
            selected_file = st.sidebar.selectbox("Select Demo File:", demo_files)
            if selected_file:
                df = load_events_data(selected_file)
        else:
            st.sidebar.warning("No CSV files found in data/output/")
    
    elif data_source == "Upload CSV":
        st.sidebar.info("💡 Tip: If file upload doesn't work, try 'Auto-load Demo' or 'File Path' options")
        uploaded_file = st.sidebar.file_uploader("Choose events CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.sidebar.success("File uploaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error reading file: {e}")
    else:
        csv_path = st.sidebar.text_input("CSV File Path", "data/output/events_09_20250823_210723.csv")
        if csv_path and os.path.exists(csv_path):
            df = load_events_data(csv_path)
            st.sidebar.success("File loaded successfully!")
        elif csv_path:
            st.sidebar.error("File not found")
            # Show available files
            if os.path.exists("data/output"):
                available_files = [f for f in os.listdir("data/output") if f.endswith('.csv')]
                if available_files:
                    st.sidebar.write("Available files:")
                    for file in available_files[:5]:  # Show first 5 files
                        st.sidebar.write(f"- data/output/{file}")
    
    if df.empty:
        st.info("Please select a data source to begin")
        st.write("### 🚀 Quick Start:")
        st.write("1. Use **Auto-load Demo** to select from processed files")
        st.write("2. Use **File Path** and enter: `data/output/events_demo_abandonment_20250824_033500.csv`")
        st.write("3. Or try **Upload CSV** if your browser supports file upload")
        return
    
    # Enhanced Filters Section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Filters & Controls")
    
    filters = {}
    
    # Event type filter
    event_types = df['type'].unique().tolist() if 'type' in df.columns else []
    st.sidebar.markdown("**Event Types:**")
    filters['event_types'] = st.sidebar.multiselect(
        "Select events to display",
        event_types,
        default=event_types,
        help="Choose which event types to show in the dashboard"
    )
    
    # Set default values for removed filters
    filters['score_range'] = (0, 100)
    filters['time_range'] = (0, 1000)
    
    # Main content
    display_alert_summary(df)
    
    # Premium Timeline
    plot_timeline(df)
    
    # Apply filters and display data
    filtered_df = apply_filters(df, filters)
    
    # Show alert images if available
    if 'image_path' in filtered_df.columns:
        display_alert_images(filtered_df, "")
    
    # (Export functionality removed as per request)


if __name__ == "__main__":
    main()
