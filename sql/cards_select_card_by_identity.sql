SELECT card_id FROM cards
WHERE card_collection_number = %(card_collection_number)s
  AND card_set_code = %(card_set_code)s
  AND card_language = %(card_language)s;