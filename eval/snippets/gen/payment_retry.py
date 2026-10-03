"""
Payment retry scheduler for home-goods e-commerce.
Input: failed_payments (schema.failed_payments), retry_config (schema.retry_config)
Output: scheduled_retries (schema.scheduled_retries)
"""
import datetime
import math

def schedule_retries(db, now=None):
    if now is None:
        now = datetime.datetime.utcnow()
    config = db.fetch_one("SELECT max_retries, base_delay_minutes, cap_per_day FROM schema.retry_config")
    max_retries, base_delay, cap = config

    failed = db.fetch_all("""
        SELECT fp.payment_id, fp.customer_id, fp.amount, fp.attempts
        FROM schema.failed_payments fp
        WHERE fp.status = 'retry_scheduled' OR fp.attempts = 0
    """)

    for row in failed:
        attempts = row['attempts'] + 1
        if attempts > max_retries:
            db.execute("UPDATE schema.failed_payments SET status = 'failed_permanent' WHERE payment_id = %s", (row['payment_id'],))
            continue

        # Exponential backoff: base_delay * 2^(attempts-1) minutes
        delay_minutes = base_delay * (2 ** (attempts - 1))
        scheduled_time = now + datetime.timedelta(minutes=delay_minutes)

        # Enforce daily cap: only schedule if fewer than 'cap' retries scheduled for today
        today = now.date()
        retry_count = db.fetch_one("""
            SELECT COUNT(*) FROM schema.scheduled_retries
            WHERE DATE(scheduled_at) = %s
        """, (today,))[0]

        if retry_count < cap:
            db.execute("""
                INSERT INTO schema.scheduled_retries (payment_id, scheduled_at, attempts)
                VALUES (%s, %s, %s)
            """, (row['payment_id'], scheduled_time, attempts))
            db.execute("""
                UPDATE schema.failed_payments SET attempts = %s, status = 'retry_scheduled'
                WHERE payment_id = %s
            """, (attempts, row['payment_id']))
        else:
            db.execute("UPDATE schema.failed_payments SET status = 'retry_cap_reached' WHERE payment_id = %s", (row['payment_id'],))
