from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Unit Converter API! Use /convert?value=10&from_unit=cm&to_unit=m"

@app.route('/convert')
def convert():
    try:
        value = float(request.args.get('value'))
        from_unit = request.args.get('from_unit')
        to_unit = request.args.get('to_unit')

        # Length conversions (base: meters)
        conversions = {
            "m": 1,
            "cm": 0.01,
            "mm": 0.001,
            "km": 1000
        }

        if from_unit not in conversions or to_unit not in conversions:
            return jsonify({"error": "Invalid units. Use m, cm, mm, or km"}), 400

        # Convert to meters first
        value_in_meters = value * conversions[from_unit]

        # Convert to target unit
        result = value_in_meters / conversions[to_unit]

        return jsonify({
            "input_value": value,
            "from_unit": from_unit,
            "to_unit": to_unit,
            "converted_value": result
        })

    except (TypeError, ValueError):
        return jsonify({"error": "Invalid input. Provide value, from_unit, and to_unit."}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)