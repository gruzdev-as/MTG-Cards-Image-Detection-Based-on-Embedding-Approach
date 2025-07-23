INSERT INTO cards (
    card_name, card_collection_number, card_set_code, 
    card_language, card_rarity, card_manacost,
    card_cmc, card_colors, card_types, card_artist
)
VALUES (
    %(card_name)s, %(card_collection_number)s, %(card_set_code)s, 
    %(card_language)s, %(card_rarity)s, %(card_manacost)s,
    %(card_cmc)s, %(card_colors)s, %(card_types)s, %(card_artist)s
)
ON CONFLICT (card_collection_number, card_set_code, card_language)
DO NOTHING
RETURNING card_id;