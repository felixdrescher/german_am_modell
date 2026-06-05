import sqlite3
import argparse
import sys

def execute_sql(db_path, statement):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Execute the provided SQL
        cursor.execute(statement)
        conn.commit()
        
        print(f"Success! Rows affected: {cursor.rowcount}")
        conn.close()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute SQL on your local DB.")
    parser.add_argument("sql", type=str, help="The SQL statement to execute (e.g., 'DELETE FROM table WHERE id=1')")
    parser.add_argument("--db", type=str, default="app/progress.db", help="Path to your .db file")
    
    args = parser.parse_args()
    execute_sql(args.db, args.sql)