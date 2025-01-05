import os
import sys
import datetime
import gzip
import requests
import sqlite3
import multiprocessing
from multiprocessing import Pool
from io import BytesIO
from PIL import Image
import tqdm

# Color mappings
COLOR_STR_TO_ID = {
    "#6D001A": 0,
    "#BE0039": 1,
    "#FF4500": 2,
    "#FFA800": 3,
    "#FFD635": 4,
    "#FFF8B8": 5,
    "#00A368": 6,
    "#00CC78": 7,
    "#7EED56": 8,
    "#00756F": 9,
    "#009EAA": 10,
    "#00CCC0": 11,
    "#2450A4": 12,
    "#3690EA": 13,
    "#51E9F4": 14,
    "#493AC1": 15,
    "#6A5CFF": 16,
    "#94B3FF": 17,
    "#811E9F": 18,
    "#B44AC0": 19,
    "#E4ABFF": 20,
    "#DE107F": 21,
    "#FF3881": 22,
    "#FF99AA": 23,
    "#6D482F": 24,
    "#9C6926": 25,
    "#FFB470": 26,
    "#000000": 27,
    "#515252": 28,
    "#898D90": 29,
    "#D4D7D9": 30,
    "#FFFFFF": 31,
}

COLOR_ID_TO_RGB = {
    0: (109, 0, 26),
    1: (190, 0, 57),
    2: (255, 69, 0),
    3: (255, 168, 0),
    4: (255, 214, 53),
    5: (255, 248, 184),
    6: (0, 163, 104),
    7: (0, 204, 120),
    8: (126, 237, 86),
    9: (0, 117, 111),
    10: (0, 158, 170),
    11: (0, 204, 192),
    12: (36, 80, 164),
    13: (54, 144, 234),
    14: (81, 233, 244),
    15: (73, 58, 193),
    16: (106, 92, 255),
    17: (148, 179, 255),
    18: (129, 30, 159),
    19: (180, 74, 192),
    20: (228, 171, 255),
    21: (222, 16, 127),
    22: (255, 56, 129),
    23: (255, 153, 170),
    24: (109, 72, 47),
    25: (156, 105, 38),
    26: (255, 180, 112),
    27: (0, 0, 0),
    28: (81, 82, 82),
    29: (137, 141, 144),
    30: (212, 215, 217),
    31: (255, 255, 255),
}

def create_progress_bar(total, download=False):
    if download:
        return tqdm.tqdm(total=total, unit='B', unit_scale=True, desc="Downloading")
    else:
        return tqdm.tqdm(total=total, desc="Processing")

def download_dataset():
    url = "https://placedata.reddit.com/data/canvas-history/2022_place_canvas_history.csv.gzip"
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    progress_bar = create_progress_bar(total_size, download=True)
    
    with open("2022_place_canvas_history.csv.gzip", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            progress_bar.update(len(chunk))
    progress_bar.close()
    
    with gzip.open("2022_place_canvas_history.csv.gzip", "rb") as f_in:
        with open("data.csv", "wb") as f_out:
            f_out.write(f_in.read())
    os.remove("2022_place_canvas_history.csv.gzip")
    print("Dataset downloaded and extracted successfully.")

def process_unsorted(canvas_size):
    conn = sqlite3.connect("unsorted.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_id TEXT UNIQUE
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pixels (
            timestamp INTEGER,
            user_id INTEGER,
            color_id INTEGER,
            x INTEGER,
            y INTEGER,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    
    with open("data.csv", "r") as f:
        lines = f.readlines()
        total_lines = len(lines)
        progress_bar = create_progress_bar(total_lines)
        
        chunk_size = 100000
        for i in range(0, total_lines, chunk_size):
            chunk = lines[i:i+chunk_size]
            if i == 0 and chunk[0].startswith("timestamp"):
                chunk = chunk[1:]
            insert_users = []
            insert_pixels = []
            for line in chunk:
                fields = line.strip().split(',')
                if len(fields) < 5:
                    continue
                x = int(fields[3].strip('"'))
                y = int(fields[4].strip('"'))
                if 0 <= x < canvas_size and 0 <= y < canvas_size:
                    timestamp_str = fields[0].strip('"')
                    try:
                        timestamp = int(datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f UTC").timestamp())
                    except ValueError:
                        timestamp = int(datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S UTC").timestamp())
                    color_str = fields[2].strip('"')
                    color_id = COLOR_STR_TO_ID.get(color_str, 31)
                    insert_users.append((fields[1].strip('"'),))
                    insert_pixels.append((timestamp, fields[1].strip('"'), color_id, x, y))
            cursor.executemany("INSERT OR IGNORE INTO users (original_id) VALUES (?)", insert_users)
            cursor.executemany("""
                INSERT INTO pixels (timestamp, user_id, color_id, x, y)
                VALUES (?, (SELECT id FROM users WHERE original_id = ?), ?, ?, ?)
            """, insert_pixels)
            conn.commit()
            progress_bar.update(len(chunk))
        progress_bar.close()
    conn.close()
    print("Unsorted database created successfully.")

def process_sorted():
    src_conn = sqlite3.connect("unsorted.db")
    dest_conn = sqlite3.connect("sorted.db")
    dest_cursor = dest_conn.cursor()
    dest_cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_id TEXT UNIQUE
        )
    """)
    dest_cursor.execute("""
        CREATE TABLE IF NOT EXISTS pixels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER,
            user_id INTEGER,
            color_id INTEGER,
            x INTEGER,
            y INTEGER,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    dest_conn.commit()
    
    src_cursor = src_conn.cursor()
    src_cursor.execute("SELECT id, original_id FROM users")
    users = src_cursor.fetchall()
    dest_cursor.executemany("INSERT INTO users (id, original_id) VALUES (?, ?)", users)
    dest_conn.commit()
    
    src_cursor.execute("SELECT COUNT(*) FROM pixels")
    total_pixels = src_cursor.fetchone()[0]
    progress_bar = create_progress_bar(total_pixels)
    
    src_cursor.execute("SELECT timestamp, user_id, color_id, x, y FROM pixels ORDER BY timestamp ASC")
    rows = src_cursor.fetchall()
    dest_cursor.executemany("""
        INSERT INTO pixels (timestamp, user_id, color_id, x, y)
        VALUES (?, ?, ?, ?, ?)
    """, rows)
    dest_conn.commit()
    progress_bar.update(len(rows))
    progress_bar.close()
    dest_conn.close()
    src_conn.close()
    print("Sorted database created successfully.")

def partition_sorted_db():
    src_conn = sqlite3.connect("sorted.db")
    src_cursor = src_conn.cursor()
    src_cursor.execute("SELECT COUNT(*) FROM pixels")
    total_pixels = src_cursor.fetchone()[0]
    partition_size = 1_000_000
    num_partitions = (total_pixels + partition_size - 1) // partition_size
    progress_bar = create_progress_bar(num_partitions)
    
    for partition in range(num_partitions):
        db_name = f"partitions/partition_{partition}.db"
        os.makedirs(os.path.dirname(db_name), exist_ok=True)
        dest_conn = sqlite3.connect(db_name)
        dest_cursor = dest_conn.cursor()
        dest_cursor.execute("""
            CREATE TABLE IF NOT EXISTS pixels (
                id INTEGER PRIMARY KEY,
                timestamp INTEGER,
                user_id INTEGER,
                color_id INTEGER,
                x INTEGER,
                y INTEGER
            )
        """)
        dest_cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON pixels (timestamp)")
        dest_cursor.execute("CREATE INDEX IF NOT EXISTS idx_coordinates ON pixels (x, y)")
        dest_conn.commit()
        
        start_id = partition * partition_size
        end_id = (partition + 1) * partition_size
        src_cursor.execute("""
            SELECT id, timestamp, user_id, color_id, x, y
            FROM pixels
            WHERE id >= ? AND id < ?
            ORDER BY id ASC
        """, (start_id, end_id))
        rows = src_cursor.fetchall()
        dest_cursor.executemany("""
            INSERT INTO pixels (id, timestamp, user_id, color_id, x, y)
            VALUES (?, ?, ?, ?, ?, ?)
        """, rows)
        dest_conn.commit()
        dest_conn.close()
        progress_bar.update(1)
    progress_bar.close()
    src_conn.close()
    print("Database partitioning complete.")

def process_end_states():
    partition_folder = "partitions"
    output_folder = "end_states"
    canvas_size = 2000
    os.makedirs(output_folder, exist_ok=True)
    end_state = [[31 for _ in range(canvas_size)] for _ in range(canvas_size)]
    
    partition_count = len([f for f in os.listdir(partition_folder) if f.startswith("partition_")])
    progress_bar = create_progress_bar(partition_count)
    
    for partition in range(partition_count):
        db_name = f"{partition_folder}/partition_{partition}.db"
        if not os.path.exists(db_name):
            continue
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT x, y, color_id FROM pixels ORDER BY timestamp ASC")
        rows = cursor.fetchall()
        for x, y, color_id in rows:
            if 0 <= x < canvas_size and 0 <= y < canvas_size:
                end_state[y][x] = color_id
        conn.close()
        with open(f"{output_folder}/end_state_{partition}.txt", "w") as f:
            for row in end_state:
                f.write(";".join(map(str, row)) + "\n")
        progress_bar.update(1)
    progress_bar.close()
    print("End states created successfully.")

def process_end_states_to_png():
    input_folder = "end_states"
    output_folder = "png_output"
    canvas_size = 2000
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]
    progress_bar = create_progress_bar(len(files))
    
    for file_name in files:
        with open(f"{input_folder}/{file_name}", "r") as f:
            lines = f.readlines()
            img = Image.new('RGB', (canvas_size, canvas_size))
            pixels = img.load()
            for y, line in enumerate(lines):
                if y >= canvas_size:
                    break
                color_ids = line.strip().split(';')
                for x, color_id_str in enumerate(color_ids):
                    if x >= canvas_size:
                        break
                    color_id = int(color_id_str)
                    color = COLOR_ID_TO_RGB.get(color_id, (0, 0, 0))
                    pixels[x, y] = color
            img.save(f"{output_folder}/{file_name.replace('.txt', '.png')}")
        progress_bar.update(1)
    progress_bar.close()
    print("PNG files created successfully.")

def main():
    mode = input("Enter the mode (download, unsorted, sorted, partition, end_states, all, png): ").strip()
    canvas_size = int(input("Enter the canvas size (2000): ") or "2000")
    
    if mode == "download":
        download_dataset()
    elif mode == "unsorted":
        process_unsorted(canvas_size)
    elif mode == "sorted":
        process_sorted()
    elif mode == "partition":
        partition_sorted_db()
    elif mode == "end_states":
        process_end_states()
    elif mode == "png":
        process_end_states_to_png()
    elif mode == "all":
        download_dataset()
        process_unsorted(canvas_size)
        process_sorted()
        partition_sorted_db()
        process_end_states()
        process_end_states_to_png()
    else:
        print("Invalid mode. Use 'download', 'unsorted', 'sorted', 'partition', 'end_states', 'all', or 'png'")

if __name__ == "__main__":
    main()