CREATE TABLE public.users (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE public.orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES public.users(id),
    amount NUMERIC NOT NULL
);
