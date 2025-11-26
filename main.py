import asyncio
import json
import os, datetime, time
from notifications import authenticate_for_gmail_notifications, create_message_with_attachment,\
    generate_notification
from logger import logger, root
from functions import test_result_w_binance_data, remove_ticker_not_in_binance, convert_columns_to_numeric
import pandas as pd
from multiprocessing import Queue, Process
import warnings
#from bybit import client, create_position_with_indicator, iterative_check_current_position_finalized, assert_position_closure, conform_predictions_for_futures
# from dydx_connection import client, create_position_with_indicator, \
#     iterative_check_current_position_finalized, assert_position_closure, conform_predictions_for_futures
from dydx_v4_connection import create_position_with_indicator, iterative_check_current_position_finalized, assert_position_closure, conform_predictions_for_futures
warnings.filterwarnings("ignore")


def generate_structure_consumer(prediction, i=0):

    gmail_service = authenticate_for_gmail_notifications()
    last_prediction = None

    try:

        if os.path.exists(root + "/last_prediction.txt"):
            with open(root + "/last_prediction.txt", 'r') as fr:
                last_prediction = fr.readlines()
                last_prediction = float(last_prediction[0])

        mytype, conformed_prediction = conform_predictions_for_futures(prediction=prediction)
        conformed_prediction = -conformed_prediction if mytype=='s' else conformed_prediction

        if last_prediction is not None and abs(round(conformed_prediction,2) - round(last_prediction,2)) <= 0.03:
            logger.info("Threshold not passed...")
            return
        elif pd.isna(conformed_prediction):
            logger.info("Nan prediction")
            return

        response = asyncio.run(iterative_check_current_position_finalized(function=assert_position_closure))
        logger.info(f"RESPONSE FROM RESET TRADE STATUS: {response}")
        position = asyncio.run(create_position_with_indicator(prediction=prediction))

        with open(root+"/last_prediction.txt", 'w+') as fw:
            fw.writelines(str(conformed_prediction))

        logger.info(
            f"PREDICTION {prediction} - CONF.PREDICTION {conformed_prediction} - LAST PREDICTION {last_prediction}"
            f"- EXECUTED NEW ORDER: \n{position}")

        create_message_with_attachment(gmail_service, 'me',
                                       message_text=f'YoBOT FUTURES - YoBOT NEW POSITION: {position}')

    except Exception as err:
        logger.warning(err)
        return err

    i += 1
    time.sleep(30)
    return


if __name__ == '__main__':

    i = 1
    start_time = time.time()
    i=0
    last_time = 0
    
    while True:
        try:
            mytime = time.time()
            df = remove_ticker_not_in_binance(["BTC"], interval="1d")
            df = pd.DataFrame.from_records(df["BTC"])
            df = convert_columns_to_numeric(df)
            current_time = float(df['BTC_open'].tail(1))
            if last_time != current_time:
                indicator_value = test_result_w_binance_data(df)
                generate_structure_consumer(indicator_value)
                last_time = current_time

            time.sleep(round(60 - (time.time()-mytime)) if (time.time()-mytime)<60-1 else 1)

        except Exception as error:

            try:
                gmail_service = authenticate_for_gmail_notifications()
                create_message_with_attachment(gmail_service, 'me',
                                           message_text=f'YoBOT FUTURES - YoBOT TDX CRASHED 0.0 - ERROR: {error}')
                print(f"YoBOT FUTURES - YoBOT TDX CRASHED 0.0 - ERROR: {error} - {datetime.datetime.strftime(datetime.datetime.today() , '%d/%m/%Y-%H:%M')}")
                logger.error(f"YoBOT FUTURES - YoBOT CRASHED 0.0 - ERROR: {error} - {datetime.datetime.strftime(datetime.datetime.today() , '%d/%m/%Y-%H:%M')}")
                time.sleep(round(300 - (time.time()-mytime)) if (time.time()-mytime)<300 else 1)
            except:
                continue
            continue
