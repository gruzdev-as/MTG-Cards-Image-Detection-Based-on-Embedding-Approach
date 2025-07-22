import logging

import psycopg2
from logging_config import setup_logging

setup_logging()
db_logger = logging.getLogger("db_logger")


class PGDBconnector:
    """Establish connect to the PG database."""

    def __init__(self, connection_params: dict) -> None:
        """Connect to the PG database to insert data in."""
        self.conn = psycopg2.connect(**connection_params)
        self.cursor = self.conn.cursor()
        print("PG connections has established")

    def add_card(self, card_data: tuple[str | int]) -> None:
        """Insert card date into the database."""
        insert_query = """
            INSERT INTO cards (card_name, card_collection_number, card_set_name,
                            card_set_code, card_language, card_rarity,
                            card_manacost, card_cmc, card_colors)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
        try:
            self.cursor.execute(insert_query, card_data)
            self.conn.commit()
            db_logger.info("Card '%s' added to the cards table.", card_data[0])
        except Exception:
            self.conn.rollback()
            db_logger.exception("Error adding card %s:", card_data[0])

    def update_or_create_inventory(self, card_data: list[tuple]) -> None:
        """Update info in the database."""
        check_query = """
        SELECT quantity FROM card_inventory
        WHERE card_id = %s AND card_condition = %s AND foil_flag = %s;
        """
        update_query = """
        UPDATE card_inventory
        SET quantity = %s, added_date = CURRENT_TIMESTAMP
        WHERE card_id = %s AND card_condition = %s AND foil_flag = %s;
        """
        insert_query = """
        INSERT INTO card_inventory (card_id, card_condition, foil_flag, quantity)
            VALUES (%s, %s, %s, %s);
        """

        for card_id, card_condition, foil_flag, quantity in card_data:
            try:
                self.cursor.execute(check_query, (card_id, card_condition, foil_flag))
                existing_inventory = self.cursor.fetchone()
                if existing_inventory:
                    self.cursor.execute(update_query, (quantity, card_id, card_condition, foil_flag))
                    db_logger.info(
                        "Updated inventory for card_id %s (Condition: %s, Foil: %s) to %s.",
                        card_id, card_condition, foil_flag, quantity,
                    )
                else:
                    self.cursor.execute(insert_query, (card_id, card_condition, foil_flag, quantity))
                    db_logger.info(
                        "Created inventory record for card_id %s (Condition: %s, Foil: %s) with quantity %s.",
                        card_id, card_condition, foil_flag, quantity,
                    )
                self.conn.commit()
            except Exception:  # noqa: PERF203 since we want to process each transaction
                self.conn.rollback()
                db_logger.exception(
                    "Error updating or creating inventory for card_id %s (Condition: %s, Foil: %s)",
                    card_id, card_condition, foil_flag,
                )

    def close_connection(self) -> None:
        """Close the connection and cursor."""
        self.cursor.close()
        self.conn.close()
        db_logger.info("Database connection closed.")
