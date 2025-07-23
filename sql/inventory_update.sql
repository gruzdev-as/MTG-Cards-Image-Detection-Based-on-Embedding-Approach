UPDATE card_inventory
SET quantity = %s, updated_date = CURRENT_TIMESTAMP
WHERE card_id = %s AND card_condition = %s AND foil_type = %s;