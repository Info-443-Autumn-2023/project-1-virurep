import pytest
from stock_processing import stocks_processing, main
from stock import Stock
import datetime as dt
import os

def test_stocks_processing(tmp_path, monkeypatch):
    tmp_dir = tmp_path / "temp_dir"
    tmp_dir.mkdir()
    temp_file = tmp_dir / "temp_stocks.txt"
    with open(temp_file, 'w') as f:
        f.write("AAPL\nGOOG\nMSFT")
    
    def mock_run_models(self):
        pass
    
    def mock_plot_predicted_vs_actual(self):
        pass
    
    def mock_plot_future(self):
        pass

    monkeypatch.setattr(Stock, 'run_models', mock_run_models)
    monkeypatch.setattr(Stock, 'plot_predicted_vs_actual', mock_plot_predicted_vs_actual)
    monkeypatch.setattr(Stock, 'plot_future', mock_plot_future)
    
    stocks_processing(filename=str(temp_file))
    
    assert Stock.run_models.call_count == 3
    assert Stock.plot_predicted_vs_actual.call_count == 3
    assert Stock.plot_future.call_count == 3

def test_main(tmp_path, monkeypatch):
    tmp_dir = tmp_path / "temp_dir"
    tmp_dir.mkdir()
    temp_file = tmp_dir / "temp_stocks.txt"
    with open(temp_file, 'w') as f:
        f.write("AAPL\nGOOG\nMSFT")
    
    main()
    
    plots_folder = tmp_dir / "plots"
    assert os.path.isdir(plots_folder)

if __name__ == '__main__':
    pytest.main()
