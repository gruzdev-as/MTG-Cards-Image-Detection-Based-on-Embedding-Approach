CREATE TABLE cards (
    card_id SERIAL PRIMARY KEY,
    card_name VARCHAR(255) NOT NULL,
    card_collection_number INT NOT NULL,
    card_set_code VARCHAR(10) NOT NULL,
    card_language CHAR(5) NOT NULL,
    card_rarity VARCHAR(10) NOT NULL, 
    card_manacost VARCHAR(255), 
    card_cmc INT NOT NULL,
    card_colors TEXT[],
    card_types TEXT[] NOT NULL,
    card_artist VARCHAR(255) NOT NULL,
    UNIQUE (card_collection_number, card_set_code, card_language)
); 

CREATE TABLE card_inventory (
    inventory_id SERIAL PRIMARY KEY,
    card_id INT NOT NULL REFERENCES cards(card_id),
    card_condition VARCHAR(10) NOT NULL,
    foil_type VARCHAR(15) NOT NULL,
    quantity INT NOT NULL CHECK (quantity >= 0),
    added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (card_id, card_condition, foil_type)
);

CREATE TABLE trades (
    trade_id SERIAL PRIMARY KEY,
    trade_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    trade_type VARCHAR(10) NOT NULL CHECK (trade_type IN ('buy', 'sell')),
    trade_price_rub NUMERIC(10, 2) NOT NULL,
    trade_notes TEXT
);

CREATE TABLE trade_details (
    trade_detail_id SERIAL PRIMARY KEY,
    trade_id INT NOT NULL REFERENCES trades(trade_id),
    inventory_id INT NOT NULL REFERENCES card_inventory(inventory_id) ON DELETE RESTRICT,
    quantity INT NOT NULL CHECK (quantity > 0),
    UNIQUE (trade_id, inventory_id)
);