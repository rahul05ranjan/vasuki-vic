"""What training worlds are about, and what they must stay away from."""

# Table names used by the eval sets. A training world may not use any of them, and the topics below
# avoid the eval sets' subjects, so the eval measures schemas the model has never seen.
BANNED_TABLES = frozenset({
    "orders", "customers", "employees", "departments", "handoffs", "readings", "shipments", "carriers",
    "students", "requests", "stock", "stock_raw", "subscriptions", "plans", "expenses", "trips", "tickets",
    # evals/test1
    "vending_machines", "vends", "elevators", "callouts", "lines", "usage", "desk_bookings", "roasts",
    "tank_tests", "hives", "hive_checks", "entries", "scores", "repairs", "animals", "cloud_costs", "found_items",
})

TOPICS = (
    "hospital outpatient appointments", "library book loans", "cricket match scorecards", "gym member visits",
    "restaurant table reservations", "cinema screenings and seats sold", "hotel room bookings",
    "airport flight departures", "apartment rental listings", "farm crop harvests", "music streaming plays",
    "podcast episodes", "museum exhibits and visitors", "marathon race results", "clinic lab test results",
    "pharmacy prescriptions", "car service appointments", "real estate sales", "bank loan applications",
    "insurance claims", "conference talks", "job applications", "freelance project invoices",
    "electricity meter bills", "water tanker deliveries", "city bus route timetables", "train ticket bookings",
    "parking garage sessions", "zoo animal feeding logs", "veterinary clinic visits", "plant nursery sales",
    "bakery daily production", "coffee shop daily sales", "wedding venue bookings", "football league fixtures",
    "chess tournament games", "video game play sessions", "mobile app store reviews", "online course enrolments",
    "language class attendance", "driving test results", "blood donation camps", "vaccination drives",
    "dental clinic treatments", "physiotherapy sessions", "ambulance call-outs", "fire station incidents",
    "police traffic fines", "court case hearings", "municipal building permits", "property tax payments",
    "voter registration drives", "charity donations", "crowdfunding campaigns", "art gallery sales",
    "auction lots", "used car listings", "bike rental stations", "electric vehicle charging sessions",
    "petrol pump daily sales", "toll plaza crossings", "courier pickup requests", "warehouse pallet movements",
    "factory machine downtime", "assembly line quality checks", "construction site material deliveries",
    "solar panel daily output", "wind turbine maintenance", "mine ore extraction shifts", "fishing boat catches",
    "dairy milk collection", "poultry farm egg counts", "tea estate plucking records", "vineyard grape harvests",
    "grain mandi auction prices", "cold storage lots", "seed distribution to farmers", "irrigation pump usage",
    "village health camp visits", "school bus routes", "college hostel room allocations", "exam hall seating",
    "scholarship applications", "research grant proposals", "laboratory equipment bookings",
    "journal paper submissions", "patent filings", "software bug bounty reports", "open source pull requests",
    "data centre rack power usage", "website A/B test variants", "email newsletter campaigns",
    "social media post engagement", "influencer brand deals", "television show ratings", "radio ad slots",
    "newspaper classified ads", "book publishing royalties", "bookshop daily sales", "stationery wholesale quotes",
    "tailor shop measurements and jobs", "laundry service loads", "salon appointments", "spa treatments",
    "yoga studio classes", "swimming pool lane bookings", "tennis court bookings", "golf course tee times",
    "ski resort lift passes", "trekking expedition permits", "safari park entries", "cruise ship cabins",
    "travel agency tour packages", "visa applications", "foreign exchange counter transactions",
    "mutual fund purchases", "stock broker trades", "credit card statements", "ATM cash refills",
    "microfinance group repayments", "chit fund contributions", "gold loan pledges", "pawn shop items",
    "event ticket scans", "music festival stage schedule", "theatre play rehearsals", "film shoot schedules",
    "photography studio bookings", "printing press jobs", "furniture workshop commissions",
    "home renovation quotes", "plumber service calls", "pest control visits", "housing society maintenance dues",
    "community hall bookings", "temple donation receipts", "blood bank units", "organ transplant waiting list",
    "space launch schedule", "satellite ground station passes", "earthquake catalogue", "river gauge levels",
    "air quality monitoring by city", "wildlife camera trap sightings", "bird ringing records",
    "beach clean-up drives", "recycling centre drop-offs", "household waste collection routes",
)

# Out of reach for the operations whatever the data: the label is `unsupported` by construction.
UNSUPPORTED_ACTIONS = (
    "draw a chart or plot of the data",
    "send an email or message to the people in the data",
    "forecast or predict future values",
    "train a machine-learning model",
    "look up live outside information, such as today's exchange rate, weather or market price",
    "translate a text column into another language",
    "find and merge rows whose names are misspellings of each other",
    "write a prose summary of what the data shows",
    "export the result as a formatted Excel workbook or PDF",
    "set up a recurring reminder or schedule",
    "do part of what the operations can do (like listing rows) together with an action they cannot do "
    "(like emailing or charting the result)",
)
