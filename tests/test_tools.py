import pytest
from src.tools import calculate_discount, calculate_final_price
def test_discount(): assert calculate_discount(1200,20)==960
def test_tax(): assert calculate_final_price(100,8.5)==108.5
@pytest.mark.parametrize("fn,args",[(calculate_discount,(-1,10)),(calculate_discount,(10,101)),(calculate_final_price,(10,-1))])
def test_invalid(fn,args):
    with pytest.raises(ValueError): fn(*args)
