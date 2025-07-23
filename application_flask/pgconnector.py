import logging
from typing import Any

import psycopg2
from logging_config import setup_logging

from utils.sql_loader import load_sql

setup_logging()
db_logger = logging.getLogger("db_logger")


class PGDBconnector:
    """Establish connect to the PG database."""

    def __init__(self, connection_params: dict) -> None:
        """Connect to the PG database to insert data in."""
        self.conn = psycopg2.connect(**connection_params)
        self.cursor = self.conn.cursor()
        db_logger.info("Connection has established")
        print("PG connection has established")

    def prepare_card_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Prepare data before inserting into cards table."""

        def _convert_colors(color_codes: list[str]) -> list[str]:
            mapping = {"W": "White", "U": "Blue", "B": "Black", "R": "Red", "G": "Green"}
            return [mapping[c] for c in color_codes if c in mapping]

        def _extract_types(type_line: str) -> list:
            return [word.strip() for word in type_line.split() if len(word) > 1]

        return {
            "card_name": data["card_name"],
            "card_collection_number": int(data["card_collector_number"]),
            "card_set_code": data["card_set"].lower(),
            "card_language": data["card_language"],
            "card_rarity": data["card_rarity"],
            "card_manacost": data["card_mana_cost"],
            "card_cmc": int(data["card_cmc"]),
            "card_colors": _convert_colors(data["card_colors"]),
            "card_types": _extract_types(data["card_type_line"]),
            "card_artist": data["card_artist"],
        }

    def add_card(self, card_data: dict[str, Any]) -> int | None:
        """Insert card date into the database."""
        insert_query = load_sql("cards_insert_card.sql")
        select_query = load_sql("cards_select_card_by_identity.sql")

        try:
            self.cursor.execute(insert_query, card_data)
            result = self.cursor.fetchone()
            if result:
                card_id = result[0]
                db_logger.info("New card '%s' added to the cards table. ID '%s'", card_data["card_name"], card_id)
            else:
                self.cursor.execute(select_query, card_data)
                card_id = self.cursor.fetchone()[0]
                db_logger.info("Existing card '%s' were found. ID '%s'", card_data["card_name"], card_id)
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            db_logger.exception("Error adding card %s:", card_data["card_name"])
            return None
        return card_id

    def update_or_create_inventory(self, *, card_id:str, card_condition:str, foil_type: bool) -> None:
        """Update info in the database."""
        check_query = load_sql("inventory_check.sql")
        update_query = load_sql("inventory_update.sql")
        insert_query = load_sql("inventory_insert.sql")

        try:
            self.cursor.execute(check_query, (card_id, card_condition, foil_type))
            current_quantity = self.cursor.fetchone()
            if current_quantity:
                self.cursor.execute(update_query, (current_quantity[0] + 1, card_id, card_condition, foil_type))
                db_logger.info(
                    "Updated inventory for card_id %s (Condition: %s, Foil: %s) to %s.",
                    card_id, card_condition, foil_type, current_quantity[0] + 1,
                )
            else:
                self.cursor.execute(insert_query, (card_id, card_condition, foil_type, 1))
                db_logger.info(
                    "Created inventory record for card_id %s (Condition: %s, Foil: %s)",
                    card_id, card_condition, foil_type,
                )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            db_logger.exception(
                "Error updating or creating inventory for card_id %s (Condition: %s, Foil: %s)",
                card_id, card_condition, foil_type,
            )

    def close_connection(self) -> None:
        """Close the connection and cursor."""
        self.cursor.close()
        self.conn.close()
        db_logger.info("Database connection closed.")
