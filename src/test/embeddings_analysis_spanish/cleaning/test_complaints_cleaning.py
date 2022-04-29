from typing import AnyStr

import pandas as pd
import pytest

from embeddings_analysis_spanish.cleaning.complaints_cleaning import ComplaintsCleaning


class ComplaintsCleaningMock(ComplaintsCleaning):
    def read_dataframe(self, path: AnyStr) -> pd.DataFrame:
        return pd.DataFrame(
            [
                [
                    "Credit reporting",
                    """
                       I have outdated information on my credit report that I have previously disputed                 
                    """,
                    "2141773"
                ],
                [
                    "Credit reporting",
                    """
                       I have outdated information on my credit report that I have previously disputed                 
                    """,
                    "2141773"
                ],
                [
                    "Consumer Loan",
                    """
                    I purchased a new car on XXXX XXXX.
                    The car dealer called Citizens Bank to get a 10 day payoff on 
                    my loan, good till XXXX XXXX. The dealer sent the check the next day. When I balanced my checkbook
                     on XXXX XXXX. I noticed that Citizens bank had taken the automatic payment out of my checking 
                     account at XXXX XXXX XXXX Bank. I called Citizens and they stated that they did not close the
                      loan until XXXX XXXX. ( stating that they did not receive the check until XXXX. XXXX. ). 
                      I told them that I did not believe that the check took that long to arrive. XXXX told me 
                      a check was issued to me for the amount overpaid, they deducted additional interest. Today 
                      ( XXXX XXXX, ) I called Citizens Bank again and talked to a supervisor named XXXX, because 
                      on XXXX XXXX. I received a letter that the loan had been paid in full ( dated XXXX, XXXX )
                       but no refund check was included. XXXX stated that they hold any over payment for 10 
                       business days after the loan was satisfied and that my check would be mailed out on Wed.
                        the XX/XX/XXXX.. I questioned her about the delay in posting the dealer payment and 
                        she first stated that sometimes it takes 3 or 4 business days to post, then she said 
                        they did not receive the check till XXXX XXXX I again told her that I did not believe
                         this and asked where is my money. She then stated that they hold the over payment for 
                         10 business days. I asked her why, and she simply said that is their policy. I asked her
                          if I would receive interest on my money and she stated no. I believe that Citizens bank
                           is deliberately delaying the posting of payment and the return of consumer 's money to
                            make additional interest for the bank. If this is not illegal it should be, it does hurt
                             the consumer and is not ethical. My amount of money lost is minimal but if they are doing 
                             this on thousands of car loans a month, then the additional interest earned for them could 
                        be staggering. I still have another car loan from Citizens Bank and I am afraid when I trade
                              that car in another year I will run into the same problem again.
                    """,
                    "2163100"
                ],
            ],
            columns=["Product", "Consumer Complaint", "Complaint ID"]
        )

    @staticmethod
    def write_dataframe(dataframe: pd.DataFrame, path: AnyStr) -> None:
        pass


@pytest.fixture
def complaints_cleaning():
    return ComplaintsCleaningMock()


def test_complaints_cleaning__features(complaints_cleaning):
    assert complaints_cleaning._ComplaintsCleaning__features[-1] == "Vehicle loan or lease"


def test_complaints_cleaning__own_stop_words(complaints_cleaning):
    assert complaints_cleaning._ComplaintsCleaning__own_stop_words[-1] == "xxxxxxxxxxxxxxxxxxx"


def test_sample_df(complaints_cleaning):
    dataframe = complaints_cleaning.read_dataframe("dummy.xlsx")
    result = complaints_cleaning.sample_df(dataframe)

    assert len(result) == 1


def test_pre_cleaning(complaints_cleaning):
    assert complaints_cleaning.pre_cleaning() is None


def test_process(complaints_cleaning):
    assert complaints_cleaning.process() is None


def test_cooking_process(complaints_cleaning):
    dataframe = complaints_cleaning.read_dataframe("dummy.xlsx")
    dataframe = complaints_cleaning.sample_df(dataframe)
    result = complaints_cleaning.cooking_process(dataframe)
    assert "XXXX" not in result.clean_complaints[2]
