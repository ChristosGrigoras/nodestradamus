CREATE VIEW public.user_orders AS
SELECT u.id, o.id AS order_id
FROM public.users u
JOIN public.orders o ON o.user_id = u.id;

CREATE MATERIALIZED VIEW public.recent_orders AS
WITH recent AS (
    SELECT * FROM public.orders
)
SELECT * FROM recent;
