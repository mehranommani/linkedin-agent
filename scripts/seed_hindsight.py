"""
Seed existing feedbacks into Hindsight
=====================================
Migrates all human feedbacks from DuckDB into Hindsight for autonomous learning.
"""
import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database import fetch_all
from backend.memory.hindsight_client import store_feedback, check_health, ensure_bank


async def seed_all_feedbacks():
    """Migrate all existing feedbacks to Hindsight."""

    # Check if Hindsight is reachable
    health = await check_health()
    if health.get("status") != "healthy":
        print(f"Hindsight server not reachable: {health.get('error')}")
        print("   Start it with: docker compose -f docker-compose.hindsight.yml up -d")
        return

    print("Hindsight server is healthy")

    # Ensure memory bank exists before seeding
    await ensure_bank()

    # Get all existing feedbacks
    print("\nFetching feedbacks from DuckDB...")
    feedbacks = fetch_all("""
        SELECT hf.id, hf.post_id, hf.rating, hf.text_feedback, hf.is_good_example,
               p.post_text, p.title, p.source
        FROM human_feedback hf
        JOIN posts p ON hf.post_id = p.id
        ORDER BY hf.created_at DESC
    """)

    if not feedbacks:
        print("No feedbacks found to migrate")
        return

    print(f"   Found {len(feedbacks)} feedbacks to migrate\n")

    # Store each in Hindsight
    success_count = 0
    for fb in feedbacks:
        feedback_id, post_id, rating, text_fb, is_good, post_text, title, source = fb

        try:
            success = await store_feedback(
                post_id=post_id,
                post_text=post_text,
                post_title=title,
                rating=rating,
                feedback_text=text_fb,
                is_good_example=is_good,
                source=source or "unknown",
            )

            if success:
                success_count += 1
                print(f"  Migrated feedback {feedback_id[:8]}: {rating} stars for post '{title[:40]}...'")
            else:
                print(f"  Failed to migrate {feedback_id[:8]}")

        except Exception as e:
            print(f"  Error migrating {feedback_id[:8]}: {e}")

    print(f"\nMigration complete: {success_count}/{len(feedbacks)} feedbacks stored in Hindsight")
    print("Hindsight will now autonomously learn from these feedbacks.")
    print("Next pipeline run will use AI-synthesized style guidance.")


if __name__ == "__main__":
    asyncio.run(seed_all_feedbacks())
