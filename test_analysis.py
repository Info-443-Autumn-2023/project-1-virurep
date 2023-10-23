import pytest
from analyze_results import write_agg_csv, main

def test_write_agg_csv(tmp_path):
    tmp_dir = tmp_path / "temp_dir"
    tmp_dir.mkdir()
    temp_file = tmp_dir / "temp_stocks.txt"
    with open(temp_file, 'w') as f:
        f.write("Sample data")
    
    write_agg_csv(input_file=str(temp_file))
    
    result_file = tmp_dir / "results.csv"
    assert result_file.is_file()

def test_main(tmp_path, monkeypatch, capsys):
    tmp_dir = tmp_path / "temp_dir"
    tmp_dir.mkdir()
    temp_file = tmp_dir / "temp_stocks.txt"
    with open(temp_file, 'w') as f:
        f.write("Sample data")
    
    monkeypatch.setattr('builtins.input', lambda _: "True")
    
    main(data=True)
    
    captured = capsys.readouterr()
    
    assert "Number of times each model was the most accurate" in captured.out
    assert "Number of times each model was the least accurate" in captured.out
    assert "Worst Model" in captured.out

if __name__ == '__main__':
    pytest.main()
