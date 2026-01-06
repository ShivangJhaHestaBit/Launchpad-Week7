import sqlite3
from pathlib import Path


DB_PATH = Path("sales.db")


def main():
    if DB_PATH.exists():
        print("Database already exists!")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE artists (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL
    );
    """)

    cursor.execute("""
    CREATE TABLE albums (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        artist_id INTEGER,
        release_year INTEGER,
        FOREIGN KEY (artist_id) REFERENCES artists(id)
    );
    """)

    cursor.execute("""
    CREATE TABLE sales (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        artist_id INTEGER,
        album_id INTEGER,
        amount REAL,
        sale_date TEXT,
        FOREIGN KEY (artist_id) REFERENCES artists(id),
        FOREIGN KEY (album_id) REFERENCES albums(id)
    );
    """)

    artists = [
        ("Coldplay",),
        ("Adele",),
        ("Ed Sheeran",),
        ("Taylor Swift",),
    ]

    cursor.executemany(
        "INSERT INTO artists (name) VALUES (?)",
        artists
    )

    albums = [
        ("Parachutes", 1, 2000),
        ("25", 2, 2015),
        ("Divide", 3, 2017),
        ("Midnights", 4, 2022),
    ]

    cursor.executemany(
        "INSERT INTO albums (title, artist_id, release_year) VALUES (?, ?, ?)",
        albums
    )

    sales = [
        (1, 1, 120000.0, "2023-01-15"),
        (1, 1, 98000.0, "2023-03-21"),
        (2, 2, 210000.0, "2023-02-10"),
        (2, 2, 185000.0, "2023-06-30"),
        (3, 3, 175000.0, "2023-04-05"),
        (4, 4, 300000.0, "2023-07-18"),
        (4, 4, 270000.0, "2023-09-09"),
    ]

    cursor.executemany(
        "INSERT INTO sales (artist_id, album_id, amount, sale_date) VALUES (?, ?, ?, ?)",
        sales
    )

    conn.commit()
    conn.close()

    print("Database created: sales.db")


if __name__ == "__main__":
    main()
