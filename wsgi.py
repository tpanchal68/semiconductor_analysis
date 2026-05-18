from src import create_app, db
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

# 1. Create the application instance
app = create_app()

# 2. Add the database creation logic here
# This only runs when run.py is executed directly (via python run.py)
with app.app_context():
    # Create database tables if they don't exist
    db.create_all()

if __name__ == '__main__':
    # Flask runs in development mode by default with auto-reloader
    # app.run(host='0.0.0.0', port=5000)
    # app.run(host='localhost', port=4700)
    app.run(host='localhost', port=4700, debug=True, threaded=False, use_reloader=False)