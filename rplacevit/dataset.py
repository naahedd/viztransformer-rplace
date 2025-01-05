import sqlite3
import numpy as np
import os
import math
import re
import pickle
from torch.utils.data import Dataset
from collections import defaultdict

class CanvasDataset(Dataset):
    """
    Base class for managing canvas datasets.
    Handles connections to SQLite partitions, final states, and user attributes.
    """
    def __init__(self, partitions_folder="partitions", final_states_folder="final_states", 
                 canvas_view_size=64, color_palette_size=32, user_category_size=8,
                 single_partition=None, partition_row_count=1000000, use_user_attributes=False, user_attributes_db=None,
                 start_x=0, start_y=0, end_x=2048, end_y=2048):
        """
        Initialize the dataset.
        
        Args:
            partitions_folder (str): Directory containing SQLite partitions.
            final_states_folder (str): Directory containing final states.
            canvas_view_size (int): Width and height of the canvas view.
            color_palette_size (int): Number of colors in the palette.
            user_category_size (int): Number of user categories.
            single_partition (int): Index of a single partition to use, or None to use all partitions.
            partition_row_count (int): Number of rows in each partition.
            use_user_attributes (bool): Whether to use user attributes.
            user_attributes_db (str): Path to the SQLite database file containing user attributes.
            start_x (int): Minimum x-coordinate of the region to use.
            start_y (int): Minimum y-coordinate of the region to use.
            end_x (int): Maximum x-coordinate of the region to use.
            end_y (int): Maximum y-coordinate of the region to use.
        """
        self.partitions_folder = partitions_folder
        self.final_states_folder = final_states_folder
        self.canvas_view_size = canvas_view_size
        self.color_palette_size = color_palette_size
        self.user_category_size = user_category_size
        self.single_partition = single_partition
        self.partition_row_count = partition_row_count
        self.use_user_attributes = use_user_attributes
        self.user_attributes_db = user_attributes_db
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y
        self.connections = {}
        self.total_entries, self.id_mapping = self._get_total_entries_and_id_mapping()
        self.final_states = []
        print("Dataset initialized.")
        
    def _get_total_entries_and_id_mapping(self):
        total = 0
        id_mapping = {}
        
        if self.single_partition is not None:
            conn = self._get_db_connection(self.single_partition)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id FROM pixels 
                WHERE x >= ? AND x < ? AND y >= ? AND y < ?
                ORDER BY id
            """, (self.start_x, self.end_x, self.start_y, self.end_y))
            for row in cursor.fetchall():
                id_mapping[total] = row[0]
                total += 1
        else:
            total_partitions = len([f for f in os.listdir(self.partitions_folder) if f.startswith("partition_") and f.endswith(".db")])
            for partition in range(total_partitions):
                conn = self._get_db_connection(partition)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id FROM pixels 
                    WHERE x >= ? AND x < ? AND y >= ? AND y < ?
                    ORDER BY id
                """, (self.start_x, self.end_x, self.start_y, self.end_y))
                for row in cursor.fetchall():
                    id_mapping[total] = (partition, row[0])
                    total += 1
        
        print(f"Total entries in selected region: {total}")
        return total, id_mapping

    def _get_total_entries(self):
        """
        Get the total number of entries in the dataset.
        
        Returns:
            int: Total number of entries.
        """
        if self.single_partition is not None:
            conn = self._get_db_connection(self.single_partition)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM pixels")
            total = cursor.fetchone()[0]
        else:
            total_partitions = len([f for f in os.listdir(self.partitions_folder) if f.startswith("partition_") and f.endswith(".db")])
            last_partition = total_partitions - 1
            conn = self._get_db_connection(last_partition)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM pixels")
            total = cursor.fetchone()[0]
            total += (total_partitions - 1) * self.partition_row_count
            
        print(f"Total entries: {total}")
        return total

    def _get_db_connection(self, partition: int) -> sqlite3.Connection:
        """
        Get a connection to the SQLite database for a partition.
        
        Args:
            partition (int): Partition number.
            
        Returns:
            sqlite3.Connection: Connection to the database.
        """
        if partition not in self.connections:
            db_path = os.path.join(self.partitions_folder, f'partition_{partition}.db')
            self.connections[partition] = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            self.connections[partition].execute("PRAGMA query_only = ON")
            self.connections[partition].execute("PRAGMA cache_size = 100000")
            self.connections[partition].execute("PRAGMA mmap_size = 10000000000")
            self.connections[partition].execute("PRAGMA journal_mode = OFF")
            self.connections[partition].execute("PRAGMA synchronous = OFF")
            self.connections[partition].execute("PRAGMA temp_store = MEMORY")
            self.connections[partition].row_factory = sqlite3.Row
        return self.connections[partition]
    
    def load_final_states(self):
        """
        Load the final states from the final_states_folder.
        """
        os.makedirs(self.final_states_folder, exist_ok=True)
        
        if self.single_partition is not None:
            final_state_file = os.path.join(self.final_states_folder, f"final_state_{self.single_partition}.txt")
            final_state = np.loadtxt(final_state_file, delimiter=";")
            print(f"Loaded final state from {final_state_file}")
            self.final_states = [final_state]
            return
                
        final_states = []
        final_state_files = sorted(
            [f for f in os.listdir(self.final_states_folder) if f.endswith(".txt")],
            key=lambda x: [int(t) if t.isdigit() else t.lower() for t in re.split('([0-9]+)', x)]
        )
        
        for final_state_file in final_state_files:
            final_state = np.loadtxt(os.path.join(self.final_states_folder, final_state_file), delimiter=";")
            final_states.append(final_state)
            print(f"Loaded final state from {final_state_file}")
        
        invalid_files = [f for f in os.listdir(self.final_states_folder) if not f.endswith(".txt")]
        for invalid_file in invalid_files:
            print(f"Invalid file in final_states_folder: {invalid_file}")
            
        self.final_states = final_states
    
    def compute_user_attributes(self, force: bool = False, chunk_size: int = 100000) -> dict:
        """
        Compute attributes for all users in the dataset using a chunk-based approach.
        
        Args:
            force (bool): Force re-computation of attributes.
            chunk_size (int): Number of rows to process in each chunk.
        
        Returns:
            dict: User attributes.
        """
        if os.path.exists('user_attributes.pkl') and not force:
            print("Loading existing user attributes...")
            with open('user_attributes.pkl', 'rb') as f:
                self.user_attributes = pickle.load(f)
            self.user_attributes_size = next(iter(self.user_attributes.values())).shape[0]
            return self.user_attributes

        print("Computing user attributes...")

        user_data = defaultdict(lambda: {
            'x_sum': 0, 'y_sum': 0,
            'x_sq_sum': 0, 'y_sq_sum': 0,
            'timestamp_sum': 0,
            'count': 0,
            'colors': np.zeros(32, dtype=int)
        })

        total_entries = self._get_total_entries()
        processed_entries = 0

        min_timestamp = float('inf')
        max_timestamp = float('-inf')

        for partition in range(len([f for f in os.listdir(self.partitions_folder) if f.startswith("partition_") and f.endswith(".db")])):
            conn = self._get_db_connection(partition)
            cursor = conn.cursor()

            offset = 0
            while True:
                cursor.execute('''
                    SELECT user_id, color_id, x, y, timestamp
                    FROM pixels
                    LIMIT ? OFFSET ?
                ''', (chunk_size, offset))
                
                chunk = cursor.fetchall()
                if not chunk:
                    break

                for row in chunk:
                    user_id, color_id, x, y, timestamp = row
                    user = user_data[user_id]
                    user['x_sum'] += x
                    user['y_sum'] += y
                    user['x_sq_sum'] += x * x
                    user['y_sq_sum'] += y * y
                    user['timestamp_sum'] += timestamp
                    user['count'] += 1
                    user['colors'][color_id] += 1
                    min_timestamp = min(min_timestamp, timestamp)
                    max_timestamp = max(max_timestamp, timestamp)

                offset += chunk_size
                processed_entries += len(chunk)
                print(f"Processed {processed_entries}/{total_entries} entries ({processed_entries/total_entries*100:.2f}%)")

        print("Computing final user attributes...")
        
        user_ids = np.array(list(user_data.keys()))
        x_sums = np.array([data['x_sum'] for data in user_data.values()])
        y_sums = np.array([data['y_sum'] for data in user_data.values()])
        x_sq_sums = np.array([data['x_sq_sum'] for data in user_data.values()])
        y_sq_sums = np.array([data['y_sq_sum'] for data in user_data.values()])
        timestamp_sums = np.array([data['timestamp_sum'] for data in user_data.values()])
        counts = np.array([data['count'] for data in user_data.values()])
        colors = np.array([data['colors'] for data in user_data.values()])

        x_means = x_sums / counts
        y_means = y_sums / counts
        x_stds = np.sqrt(x_sq_sums / counts - x_means ** 2)
        y_stds = np.sqrt(y_sq_sums / counts - y_means ** 2)
        avg_timestamps = timestamp_sums / counts
        color_hists = colors / counts[:, np.newaxis]

        x_means_norm = x_means / 1000
        y_means_norm = y_means / 1000
        x_stds_norm = x_stds / 1000
        y_stds_norm = y_stds / 1000
        counts_norm = np.minimum(counts / 1000, 1.0)
        timestamp_norm = (avg_timestamps - min_timestamp) / (max_timestamp - min_timestamp)

        attributes = np.column_stack([
            x_means_norm, y_means_norm, x_stds_norm, y_stds_norm,
            counts_norm, timestamp_norm, color_hists
        ])

        self.user_attributes = dict(zip(user_ids, attributes))
        self.user_attributes_size = attributes.shape[1]

        with open('user_attributes.pkl', 'wb') as f:
            pickle.dump(self.user_attributes, f)

        print(f"Computed attributes for {len(self.user_attributes)} users")
        return self.user_attributes
            
    def store_user_attributes(self, db_path='user_attributes.db'):
        """
        Store user attributes in an SQLite database with user ID as the primary key.
        
        Args:
            db_path (str): Path to the SQLite database file.
        """
        print(f"Storing user attributes in {db_path}...")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_attributes (
            user_id INTEGER PRIMARY KEY,
            attributes BLOB
        )
        ''')

        data = []
        for user_id, attributes in self.user_attributes.items():
            try:
                int_user_id = int(user_id)
                serialized_attributes = pickle.dumps(attributes)
                data.append((int_user_id, sqlite3.Binary(serialized_attributes)))
            except ValueError:
                print(f"Warning: Skipping invalid user_id: {user_id}")

        cursor.executemany('INSERT OR REPLACE INTO user_attributes (user_id, attributes) VALUES (?, ?)', data)

        conn.commit()
        conn.close()

        print(f"Stored attributes for {len(data)} users in {db_path}")

        self.user_attributes_db = db_path      
        
    def get_user_attributes(self, user_id):
        """
        Retrieve user attributes for a specific user ID from the SQLite database.
        
        Args:
            user_id (int): The ID of the user whose attributes to retrieve.
        
        Returns:
            np.array: User attributes for the specified user ID, or None if not found.
        """
        if not hasattr(self, 'user_attributes_db'):
            raise ValueError("User attributes database not set. Call store_user_attributes() first.")

        if not hasattr(self, 'user_attributes_conn'):
            self.user_attributes_conn = sqlite3.connect(f"file:{self.user_attributes_db}?mode=ro", uri=True)
            self.user_attributes_conn.execute("PRAGMA query_only = ON")

        cursor = self.user_attributes_conn.cursor()
        cursor.execute('SELECT attributes FROM user_attributes WHERE user_id = ?', (int(user_id),))
        result = cursor.fetchone()

        if result:
            return pickle.loads(result[0])
        else:
            return None
    
    def _create_canvas_view(self, x, y, timestamp, user_attributes):
        """
        Helper method to create the canvas view input for a given pixel.
        
        Args:
            x (int): X-coordinate of the center pixel.
            y (int): Y-coordinate of the center pixel.
            timestamp (int): Timestamp of the pixel placement.
            user_attributes (np.array): Attributes of the user who placed the pixel.
            
        Returns:
            np.array: Canvas view input.
        """
        view_x_min = max(x - self.canvas_view_size // 2, self.start_x)
        view_x_max = min(x + self.canvas_view_size // 2, self.end_x)
        view_y_min = max(y - self.canvas_view_size // 2, self.start_y)
        view_y_max = min(y + self.canvas_view_size // 2, self.end_y)
        
        num_user_attributes = len(user_attributes)
        
        view = np.zeros((self.canvas_view_size, self.canvas_view_size, self.color_palette_size + num_user_attributes), dtype=np.float32)
        partition = self.single_partition if self.single_partition is not None else math.floor(timestamp / self.partition_row_count)
        if partition == 0:
            source_view = np.zeros((self.canvas_view_size, self.canvas_view_size), dtype=np.uint8)
        else:
            partition = min(partition, len(self.final_states) - 1)
            source_view = self.final_states[partition - 1][view_y_min:view_y_max, view_x_min:view_x_max]

        source_view_int = source_view.astype(int)
        view[:, :, source_view_int] = 1
        
        for i, attribute in enumerate(user_attributes):
            view[:, :, self.color_palette_size + i] = attribute
        
        conn = self._get_db_connection(partition)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT x, y, color_id, MAX(timestamp) as timestamp
            FROM pixels
            WHERE timestamp < ? AND x >= ? AND x < ? AND y >= ? AND y < ?
            GROUP BY x, y
        ''', (timestamp, view_x_min, view_x_max, view_y_min, view_y_max))
        
        for pixel in cursor.fetchall():
            px, py, pcolor_id, ptimestamp = pixel['x'], pixel['y'], pixel['color_id'], pixel['timestamp']
            view_x = px - view_x_min
            view_y = py - view_y_min
            view[view_y, view_x, pcolor_id] = 1
            
        return view.transpose(2, 0, 1)

    def __len__(self) -> int:
        """
        Get the total number of entries in the dataset.
        
        Returns:
            int: Total number of entries.
        """
        return self.total_entries

    def __getitem__(self, idx) -> tuple:
        """
        Get an item from the dataset.
        
        Args:
            idx (int): Index of the item.
        """
        raise NotImplementedError

    def __del__(self):
        """
        Close all connections when the object is deleted.
        """
        for conn in self.connections.values():
            conn.close()
            
class CanvasColorDataset(CanvasDataset):
    """
    Dataset for predicting the next color of the center pixel in a canvas view.
    """
    def __getitem__(self, idx) -> tuple:
        """
        Get an item from the dataset.
        Target is the one-hot encoded most probable color of the pixel in the next minute.
        
        Args:
            idx (int): Index of the item.
            
        Returns:
            tuple: (view, target)
        """
        if idx >= len(self.id_mapping):
            raise IndexError("Index out of range.")
        
        real_id = self.id_mapping[idx]
        
        if self.single_partition is not None:
            partition = self.single_partition
            pixel_id = real_id
        else:
            partition, pixel_id = real_id
        
        conn = self._get_db_connection(partition)
        cursor = conn.cursor()
        
        cursor.execute('SELECT x, y, color_id, user_id, timestamp FROM pixels WHERE id = ?', (pixel_id,))
        row = cursor.fetchone()
        x, y, color_id, user_id, timestamp = row['x'], row['y'], row['color_id'], row['user_id'], row['timestamp']
        
        user_attributes = np.zeros(0)
        if self.use_user_attributes:
            user_attributes = self.get_user_attributes(user_id)
            if user_attributes is None:
                user_attributes = np.zeros(self.user_attributes_size)
                
        view = self._create_canvas_view(x, y, timestamp, user_attributes)
        
        one_minute_later = timestamp + 60
        cursor.execute('''
            SELECT color_id, COUNT(*) as count
            FROM pixels
            WHERE x = ? AND y = ? AND timestamp > ? AND timestamp <= ?
            GROUP BY color_id
            ORDER BY count DESC
            LIMIT 1
        ''', (x, y, timestamp, one_minute_later))
        
        most_probable_color = cursor.fetchone()
        
        if most_probable_color is None:
            most_probable_color_id = color_id
        else:
            most_probable_color_id = most_probable_color['color_id']
        
        target = np.zeros(self.color_palette_size, dtype=np.uint8)
        target[most_probable_color_id] = 1
        
        target = target.astype(np.float32)
            
        return view, target

class CanvasTimeDataset(CanvasDataset):
    """
    Dataset for predicting the time before the next change of the center pixel in a canvas view.
    Allows predicting the next pixel to change before using the color transformer.
    """
    def __getitem__(self, idx) -> tuple:
        """
        Get an item from the dataset.
        Target is 16x16, with the normalized time before the next change for each pixel in the center view.

        Args:
            idx (int): Index of the item.
            
        Returns:
            tuple: (view, target)
        """
        idx = idx + 1
        partition = self.single_partition if self.single_partition is not None else math.floor(idx / self.partition_row_count)
        if self.single_partition is not None:
            idx = idx % self.partition_row_count
            idx += self.partition_row_count * partition

        conn = self._get_db_connection(partition)
        cursor = conn.cursor()

        cursor.execute('SELECT x, y, color_id, user_id, timestamp FROM pixels WHERE id = ?', (idx,))
        initial_pixel = cursor.fetchone()
        x, y, color_id, user_id, timestamp = initial_pixel['x'], initial_pixel['y'], initial_pixel['color_id'], initial_pixel['user_id'], initial_pixel['timestamp']
            
        user_attributes = np.zeros(0)
        if self.use_user_attributes:
            user_attributes = self.get_user_attributes(user_id)
            if user_attributes is None:
                user_attributes = np.zeros(self.user_attributes_size)
                
        view = self._create_canvas_view(x, y, timestamp, user_attributes)
        
        center_x_min = x - 8
        center_x_max = x + 8
        center_y_min = y - 8
        center_y_max = y + 8

        cursor.execute('''
            SELECT x, y, MIN(timestamp) as next_timestamp
            FROM pixels
            WHERE timestamp > ? AND x >= ? AND x < ? AND y >= ? AND y < ?
            GROUP BY x, y
        ''', (timestamp, center_x_min, center_x_max, center_y_min, center_y_max))
        
        target = np.ones((16, 16), dtype=np.float32)

        for pixel in cursor.fetchall():
            px, py, next_timestamp = pixel['x'], pixel['y'], pixel['next_timestamp']
            target_x = px - center_x_min
            target_y = py - center_y_min
            time_diff = next_timestamp - timestamp
            normalized_time = min(time_diff / 3600, 1.0)
            target[target_y, target_x] = normalized_time

        return view, target
            
if __name__ == "__main__":
    dataset = CanvasColorDataset(single_partition=0, start_x=0, start_y=0, end_x=256, end_y=256)
    dataset.load_final_states()
    print(len(dataset))
    print(dataset[0])
    
    #dataset = CanvasTimeDataset(single_partition=0)
    #dataset.load_final_states()
    #print(len(dataset))
    #print(dataset[0])