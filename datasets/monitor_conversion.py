#!/usr/bin/env python3
"""
Monitor PDF to Scanned PDF Conversion Progress
"""

import time
from pathlib import Path


def monitor_conversion():
    """Monitor the conversion progress"""
    input_dir = Path("../data/INFORMATION-TECHNOLOGY")
    output_dir = Path("../data/SCAN-INFORMATION-TECHNOLOGY")

    total_files = len(list(input_dir.glob("*.pdf")))

    while True:
        try:
            converted_files = len(list(output_dir.glob("SCAN_*.pdf")))
            progress = (converted_files / total_files) * 100

            print(
                f"📊 Conversion Progress: {converted_files}/{total_files} ({progress:.1f}%)"
            )

            if converted_files >= total_files:
                print("✅ Conversion completed!")
                break

            time.sleep(10)  # Check every 10 seconds

        except KeyboardInterrupt:
            print("\n⏹️ Monitoring stopped by user")
            break
        except Exception as e:
            print(f"❌ Error monitoring: {e}")
            break


if __name__ == "__main__":
    monitor_conversion()
