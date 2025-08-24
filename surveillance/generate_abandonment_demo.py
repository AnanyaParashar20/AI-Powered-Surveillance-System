#!/usr/bin/env python3
"""
Generate synthetic abandonment events for dashboard demonstration.
"""

import csv
import os
from datetime import datetime
import random

def create_abandonment_demo_data():
    """Create demo CSV with abandonment events."""
    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"{output_dir}/events_demo_abandonment_{timestamp}.csv"
    
    # CSV header
    headers = ["frame_idx", "time_sec", "type", "track_id", "score", "bbox", "contributors", "image_path"]
    
    events = []
    
    # Generate sample abandonment events
    abandonment_scenarios = [
        {
            "frame": 450, "time": 18.0, "track": "obj_24", "score": 75.5,
            "object": "backpack", "owner_track": "person_5",
            "location": [250, 180, 45, 65]
        },
        {
            "frame": 680, "time": 27.2, "track": "obj_31", "score": 82.3,
            "object": "handbag", "owner_track": "person_12", 
            "location": [380, 220, 35, 40]
        },
        {
            "frame": 920, "time": 36.8, "track": "obj_47", "score": 68.9,
            "object": "suitcase", "owner_track": "person_8",
            "location": [150, 160, 60, 80]
        }
    ]
    
    for scenario in abandonment_scenarios:
        contributor = {
            "object_track_id": scenario["track"],
            "object_type": scenario["object"],
            "owner_track_id": scenario["owner_track"],
            "static_duration": random.uniform(5.2, 12.8),
            "owner_absent_duration": random.uniform(8.5, 15.3),
            "bbox": scenario["location"],
            "center": [
                scenario["location"][0] + scenario["location"][2]/2,
                scenario["location"][1] + scenario["location"][3]/2
            ]
        }
        
        # Create image path (even though image doesn't exist)
        image_path = f"data/output/frames/demo_frame{scenario['frame']:06d}_abandonment_{scenario['track']}_{timestamp}.jpg"
        
        event = [
            scenario["frame"],
            scenario["time"],
            "abandonment",
            scenario["track"],
            scenario["score"],
            f'[{", ".join(map(str, scenario["location"]))}]',
            str([contributor]).replace("'", '"'),
            image_path
        ]
        events.append(event)
    
    # Also add some loitering events for comparison
    for i in range(3):
        frame = 300 + i * 200
        time_sec = frame / 25.0
        track_id = f"person_{10 + i}"
        score = random.uniform(45, 55)
        
        contributor = {
            "track_id": track_id,
            "dwell_seconds": random.uniform(15, 25),
            "bbox": [random.randint(100, 400), random.randint(100, 250), 
                    random.randint(25, 45), random.randint(80, 120)],
            "center": [random.randint(150, 450), random.randint(140, 290)]
        }
        
        image_path = f"data/output/frames/demo_frame{frame:06d}_loitering_{track_id}_{timestamp}.jpg"
        
        event = [
            frame, time_sec, "loitering", track_id, score,
            "[0.0, 0.0, 0.0, 0.0]",
            str([contributor]).replace("'", '"'),
            image_path
        ]
        events.append(event)
    
    # Write to CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(events)
    
    print(f"✅ Created demo abandonment data: {csv_path}")
    print(f"📊 Generated {len(abandonment_scenarios)} abandonment events")
    print(f"📊 Generated {3} loitering events for comparison")
    print(f"🎯 Load this file in the dashboard to see all event types!")
    
    return csv_path

if __name__ == "__main__":
    create_abandonment_demo_data()
