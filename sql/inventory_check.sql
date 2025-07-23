SELECT quantity FROM card_inventory
WHERE card_id = %s AND card_condition = %s AND foil_type = %s;