import asyncio
import logging
import random
import math
import time
from dydx_v4_client.key_pair import KeyPair
from dydx_v4_client.wallet import Wallet
from dydx_v4_client.node.client import NodeClient
from dydx_v4_client.network import make_mainnet, TESTNET_FAUCET
from dydx_v4_client.indexer.rest.indexer_client import IndexerClient
from dydx_v4_client.node.market import Market
from dydx_v4_client import OrderFlags
from dydx_v4_client.indexer.rest.constants import OrderType
from v4_proto.dydxprotocol.clob.order_pb2 import Order

# -----------------------------
# Configuration Constants
# -----------------------------
ORDER_LAG = 1000
ORDER_PRICE_SAFE_PERC = 0.0005
LEVERAGE = 2
MIN_FEE = 0.0005  # Default minimum fee, will be updated


# -----------------------------
# 1. Load mnemonic
# -----------------------------
def load_mnemonic(path="/home/daltonik/Desktop/futures_btc_trading/file_folder/mnemonic.txt"):
    with open(path) as f:
        return f.read().replace("\n", " ").strip()

MNEMONIC = load_mnemonic()


# -----------------------------
# 2. Connect to network
# -----------------------------
async def connect_node(test=False):
    if test:
        from dydx_v4_client.network import make_testnet
        network_config = make_testnet(
            node_url="oegs-testnet.dydx.exchange:443",
            rest_indexer="https://indexer.v4testnet.dydx.exchange",
            websocket_indexer="wss://indexer.v4testnet.dydx.exchange/v4/ws")
    else:
        network_config = make_mainnet(
            node_url="oegs.dydx.trade:443",
            rest_indexer="https://indexer.dydx.trade",
            websocket_indexer="wss://indexer.dydx.trade/v4/ws",
        )

    node = await NodeClient.connect(network_config.node)
    return node, network_config


NODE, CONFIG = asyncio.run(connect_node(test=False))
INDEXER = IndexerClient(CONFIG.rest_indexer)


# -----------------------------
# 3. Create wallet
# -----------------------------
async def create_wallet(node, mnemonic, test=False):
    key_pair = KeyPair.from_mnemonic(mnemonic)
    address = Wallet(key_pair, 0, 0).address

    print(f"Generated address: {address}")

    if test:
        try:
            from dydx_v4_client.faucet_client import FaucetClient
            print("Funding account with testnet faucet...")
            faucet = FaucetClient(TESTNET_FAUCET)
            await faucet.fill(address, 0, 2000)  # USDC
            await faucet.fill_native(address)  # Native token
            print("Faucet funding completed")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"Faucet error (might be expected): {e}")

    try:
        wallet = await Wallet.from_mnemonic(node, mnemonic, address)
        print("Wallet created successfully")
        return wallet, address
    except Exception as e:
        print(f"Error creating wallet: {e}")
        wallet = Wallet(key_pair, 0, 0)
        return wallet, address


WALLET, ADDRESS = asyncio.run(create_wallet(NODE, MNEMONIC))

# -----------------------------
# 4. Get market info
# -----------------------------
async def get_market(indexer=INDEXER, market_id="BTC-USD"):
    market_data = (await indexer.markets.get_perpetual_markets(market_id))["markets"][market_id]
    market = Market(market_data)
    return market

MARKET = asyncio.run(get_market(INDEXER, market_id="BTC-USD"))


# -----------------------------
# 5. Generate order ID
# -----------------------------
def generate_order_id(market=MARKET, address=ADDRESS):
    return market.order_id(
        address,
        0,  # subaccount number
        random.randint(0, 100000000),  # client ID
        OrderFlags.SHORT_TERM
    )


# -----------------------------
# 6. Create order
# -----------------------------
async def create_order(order_id, node=NODE, market=MARKET, size=0.01, price=1000, side=Order.Side.SIDE_BUY,
                       order_type=OrderType.LIMIT):
    good_til_block = await node.latest_block_height() + 20
    order = market.order(
        order_id,
        order_type,
        side,
        size=size,
        price=price,
        time_in_force=Order.TimeInForce.TIME_IN_FORCE_FILL_OR_KILL,  # FOK equivalent
        reduce_only=False,
        good_til_block=good_til_block,
    )
    return order, good_til_block


# -----------------------------
# 7. Cancel order
# -----------------------------
async def cancel_order(order_id, good_til_block, node=NODE, wallet=WALLET):
    return await node.cancel_order(wallet, order_id, good_til_block)


# -----------------------------
# 8. Get account information
# -----------------------------
async def get_account(wallet=WALLET, indexer=INDEXER):
    try:
        account = await indexer.account.get_subaccount(wallet.address, 0)
        return account["subaccount"]
    except Exception as e:
        print(f"Error getting account: {e}")
        return None


# -----------------------------
# 9. Get open positions
# -----------------------------
async def get_open_positions(indexer=INDEXER, wallet=WALLET):
    try:
        positions = await indexer.account.get_subaccount_perpetual_positions(wallet.address, 0)
        open_positions = []
        for position in positions.get("positions", []):
            if float(position.get("size", 0)) != 0:
                open_positions.append(position)
        return open_positions
    except Exception as e:
        print(f"Error getting positions: {e}")
        return []


# -----------------------------
# 10. Get market parameters
# -----------------------------
async def get_market_parameters(indexer=INDEXER, market_id="BTC-USD"):
    try:
        market_data = (await indexer.markets.get_perpetual_markets(market_id))["markets"][market_id]
        res = {}
        res['tick_size'] = float(market_data['tickSize'])
        res['price'] = float(market_data['oraclePrice'])
        res['step_size'] = float(market_data['stepSize'])
        #res['max_position'] = float(market_data['maxPositionSize'])
        #res['liquidation_fraction'] = float(market_data['maintenanceMarginFraction'])
        #res['liquidation_initial'] = float(market_data['initialMarginFraction'])
        #res['step_size'] = float(market_data['stepSize'])
        #res['max_leverage'] = 1 / res['liquidation_initial'] if res['liquidation_initial'] > 0 else 0
        return res
    except Exception as e:
        print(f"Error getting market parameters: {e}")
        return None


# -----------------------------
# 11. Compute safe price
# -----------------------------
async def compute_safe_price(indexer=INDEXER, side='buy', reprice_factor=ORDER_PRICE_SAFE_PERC):
    market_params = await get_market_parameters(indexer)
    if not market_params:
        return None

    oracle_price = market_params['price']
    tick_size = market_params['tick_size']

    if side == 'sell':
        price_safe = oracle_price * (1 - reprice_factor)
    else:  # buy
        price_safe = oracle_price * (1 + reprice_factor)

    # Round to tick size
    price_safe = round(price_safe / tick_size) * tick_size
    return price_safe


# -----------------------------
# 12. Conform predictions for futures
# -----------------------------
def conform_predictions_for_futures(prediction=None):
    mytype = 'buy'
    if prediction < 0.5:
        mytype = 'sell'
        prediction = 1 - prediction
    return mytype, prediction


# -----------------------------
# 13. Choose leverage
# -----------------------------
async def choose_leverage(indexer=INDEXER, wallet=WALLET, leverage=LEVERAGE, factor=None):
    market_params = await get_market_parameters(indexer)
    account = await get_account(wallet, indexer)

    if not market_params or not account:
        return 0

    equity = float(account.get('equity', 0))
    base_asset_equivalent = equity / market_params['price']
    amount_with_leverage = base_asset_equivalent * leverage

    if factor is not None:
        amount_with_leverage = amount_with_leverage * factor

    step_size = market_params['step_size']
    amount_with_leverage = round(amount_with_leverage / step_size) * step_size
    return amount_with_leverage


# -----------------------------
# 14. Close position
# -----------------------------
async def assert_position_closure(node=NODE, wallet=WALLET, indexer=INDEXER, market=MARKET, fee=MIN_FEE, reprice_factor=ORDER_PRICE_SAFE_PERC):
    try:
        positions = await get_open_positions(indexer, wallet)
        if not positions:
            return 'empty'

        position = positions[0]  # Assuming single position for simplicity
        position_size = float(position.get('size', 0))

        if position_size == 0:
            return 'empty'

        if position["status"] != "OPEN":
            logging.info("Position is not open...")
            return None

        # Determine side based on position
        side = Order.Side.SIDE_BUY if position["side"] == "LONG" else Order.Side.SIDE_SELL
        close_side = Order.Side.SIDE_BUY if side == Order.Side.SIDE_SELL else Order.Side.SIDE_SELL

        price_safe = await compute_safe_price(indexer, 'buy' if side == Order.Side.SIDE_BUY else 'sell', reprice_factor)

        order_id = generate_order_id(market, wallet.address)
        order, good_til_block = await create_order(
            order_id, node, market,
            size=abs(position_size),
            price=int(price_safe),
            side=close_side,
            order_type=OrderType.MARKET
        )

        # Place the order
        result = await node.place_order(wallet, order)
        return order_id

    except Exception as e:
        print(f"Error closing position: {e}")
        return None


# -----------------------------
# 15. Open new position
# -----------------------------
async def open_new_position(node=NODE, wallet=WALLET, indexer=INDEXER, market=MARKET, fee=MIN_FEE, side='buy',
                            order_type=OrderType.MARKET, factor=None, reprice_factor=ORDER_PRICE_SAFE_PERC):
    try:
        price_safe = await compute_safe_price(indexer, side, reprice_factor)
        logging.info(f"Leverage: {LEVERAGE}")
        size = await choose_leverage(indexer, wallet, LEVERAGE, factor)

        order_side = Order.Side.SIDE_SELL if side == 'sell' else Order.Side.SIDE_BUY

        order_id = generate_order_id(market, wallet.address)
        order, good_til_block = await create_order(
            order_id, node, market,
            size=size,
            price=int(price_safe),
            side=order_side,
            order_type=order_type
        )

        # Place the order
        result = await node.place_order(wallet, order)
        return order_id

    except Exception as e:
        print(f"Error opening position: {e}")
        return None


# -----------------------------
# 16. Create position with indicator
# -----------------------------
async def create_position_with_indicator(node=NODE, wallet=WALLET, indexer=INDEXER, market=MARKET, prediction=0.0):
    side, factor = conform_predictions_for_futures(prediction)
    order_id = await open_new_position(node, wallet, indexer, market, side=side, factor=factor)

    if order_id:
        # Wait for order to be processed
        await asyncio.sleep(2)
        # In a real implementation, you'd query the order status here
        return order_id
    return None


# -----------------------------
# 17. Iterative position check
# -----------------------------
async def iterative_check_current_position_finalized(function, node=NODE, wallet=WALLET, indexer=INDEXER, market=MARKET,
                                                     fee=MIN_FEE, reprice_factor=ORDER_PRICE_SAFE_PERC):
    order_id = await function(node, wallet, indexer, market, fee, reprice_factor=reprice_factor)

    if order_id == 'empty':
        return order_id

    attempt = 0
    max_attempts = 5

    while attempt < max_attempts:
        try:
            # Check order status (simplified - you'd need to implement proper order status checking)
            await asyncio.sleep(2)
            # In real implementation, query order status from indexer
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            reprice_factor += reprice_factor * 0.1
            fee += fee * 0.1
            order_id = await function(node, wallet, indexer, market, round(fee, 4), reprice_factor=reprice_factor)
            attempt += 1

    return order_id


# -----------------------------
# Example usage
# -----------------------------



# print("Oracle price:", market.market["oraclePrice"])
#
# # Get account info
# account = await get_account(wallet, indexer)
# print("Account equity:", account.get('equity', 'N/A'))
#
# # Get open positions
# positions = await get_open_positions(indexer, wallet)
# print("Open positions:", positions)
#
# # Example: Close position if any
# close_result = await assert_position_closure(node, wallet, indexer, market)
# print("Close position result:", close_result)
#
# # Example: Open new position with prediction
# prediction = 0.7  # Example prediction
# new_position = await create_position_with_indicator(node, wallet, indexer, market, prediction)
# print("New position ID:", new_position)
