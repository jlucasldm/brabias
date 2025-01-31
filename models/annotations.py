from datetime import datetime
from typing import Annotated
from sqlalchemy import BigInteger, Integer, func
from sqlalchemy.dialects.mysql import BIGINT, LONGTEXT, TEXT, TIMESTAMP, VARCHAR
from sqlalchemy.orm import mapped_column

MyBigInteger = BigInteger()
MyBigInteger = MyBigInteger.with_variant(Integer, "sqlite")
MyBigInteger = MyBigInteger.with_variant(BIGINT(unsigned=True), "mysql")

MyLongText = LONGTEXT()
MyLongText = MyLongText.with_variant(TEXT, "sqlite")

big_int = Annotated[int, mapped_column(MyBigInteger)]
big_intpk = Annotated[int, mapped_column(MyBigInteger, primary_key=True)]
datetime_default_now = Annotated[
    datetime, mapped_column(TEXT, default=datetime.now, server_default=func.now())
]
intpk = Annotated[int, mapped_column(Integer, primary_key=True)]
longtext = Annotated[str, mapped_column(MyLongText)]
text = Annotated[str, mapped_column(TEXT)]
timestamp = Annotated[datetime, mapped_column(TIMESTAMP)]
varchar = Annotated[str, mapped_column(VARCHAR(255))]
